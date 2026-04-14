"""
Phase XCIII-P452/453/454/455: 信号-空间频谱耦合理论
======================================================================

核心目标: 确认ratio随k增长到接近1.0的曲线，建立频谱耦合的解析框架

P452: 多尺度k重测recoding_ratio (★★★★★最高优先级★★★★★)
  - 用不同的k_svd(300/500/800/1200/1500)重测recoding_ratio
  - 预期: ratio随k增长，从0.22(k=300)增长到接近1.0(k=max)
  - 关键: ratio(k)的增长曲线形状→揭示频谱耦合的机制
  - 如果ratio(k)先快后慢→信号能量集中在大奇异值方向
  - 如果ratio(k)线性增长→信号能量均匀分布在所有奇异值方向

P453: 逃逸方向功能识别
  - P450发现逃逸能量集中在6-7个方向，这些方向对应什么?
  - 方法: 计算逃逸方向与各种"功能方向"的对齐度
    - 与LayerNorm方向的对齐度
    - 与注意力Q/K/V方向的对齐度
    - 与残差流方向的对齐度
    - 与W_U奇异值方向的对齐度(应该≈0，已验证)
  - 如果逃逸方向与LN对齐→逃逸能量被LN吸收(正则化)
  - 如果逃逸方向与Q/K对齐→逃逸能量参与注意力计算

P454: 频谱耦合函数拟合
  - 假设ratio(k) = f(信号频谱, W_U频谱, k)
  - 候选模型:
    a) ratio(k) = 1 - exp(-alpha * k) — 指数饱和
    b) ratio(k) = (k/d)^beta — 幂律
    c) ratio(k) = Σ(w_i * s_i^2) / Σ(s_i^2) — 加权求和(w_i=投影能量权重)
  - 对实验数据拟合，选出最佳模型

P455: 信号频谱与W_U频谱的统计关系
  - 计算: 信号Δ在不同奇异值方向上的投影能量 vs 奇异值大小
  - 如果投影能量与奇异值无关(均匀)→ratio(k)=k/d → 纯随机
  - 如果投影能量与奇异值正相关→信号集中在大奇异值方向
  - 如果投影能量与奇异值负相关→信号集中在小奇异值方向(反聚焦!)

实验模型: qwen3 → glm4 → deepseek7b (串行，避免GPU溢出)
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
PHASE_NAME = "Phase XCIII"
EXPERIMENT_DESC = "信号-空间频谱耦合理论"

OUT_DIR = Path("tests/glm5_temp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 测试数据
TEST_ATTRS = ["red", "blue", "big", "small", "hot", "cold", "sweet", "sour"]
PROMPT = "The apple is"
BETA = 8.0


# ================================================================
# P452: 多尺度k重测recoding_ratio
# ================================================================

def run_p452(model, tokenizer, device, model_info):
    """
    P452: 多尺度k重测recoding_ratio
    
    原理:
      Phase XCI的所有ratio测量都用k=300截断SVD，得到ratio≈0.22。
      Phase XCII发现W_U几乎满秩(PR/d≈0.89)，ratio(k=max)≈1.0。
      本实验用多个k值重测ratio，确认ratio(k)的增长曲线。
      
      ratio(k) = ||U_k^T * Delta||^2 / ||Delta||^2
      其中U_k是W_U^T的前k个奇异向量。
      
      关键问题: ratio(k)的增长曲线是什么形状?
      - 线性: ratio(k) = k/d → 信号能量均匀分布
      - 指数饱和: ratio(k) = 1 - exp(-alpha*k) → 信号集中在大SV方向
      - 幂律: ratio(k) = (k/d)^beta → 信号有长尾分布
    """
    print(f"\n{'='*60}")
    print(f"P452: 多尺度k重测recoding_ratio - {model_info.name}")
    print(f"{'='*60}")
    
    results = {}
    d_model = model_info.d_model
    n_layers = model_info.n_layers
    
    # Step 1: 预计算W_U^T的SVD (用最大k)
    W_U = get_W_U(model)
    vocab_size = W_U.shape[0]
    print(f"  W_U shape: [{vocab_size}, {d_model}]")
    
    # 预获取属性方向
    attr_directions = {}
    for attr in TEST_ATTRS:
        direction, _ = get_attr_direction(model, tokenizer, attr, W_U=W_U)
        if direction is not None:
            attr_directions[attr] = direction
    
    W_U_T = W_U.T.astype(np.float32)
    del W_U; import gc; gc.collect()
    
    max_k = min(W_U_T.shape[0], W_U_T.shape[1]) - 2
    # GLM4内存可能不够做大k，限制到1500
    if model_info.name == "glm4":
        max_k = min(max_k, 1500)
    
    # 多个k值
    k_values = [300, 500, 800, 1200, 1500, max_k]
    k_values = sorted(set(k for k in k_values if k <= max_k))
    
    print(f"  k值: {k_values}, max_k={max_k}")
    
    # 对每个k做SVD并测量ratio
    all_ratio_curves = {}  # {attr: {layer: {k: ratio}}}
    
    for k in k_values:
        print(f"\n  SVD k={k}...")
        try:
            U_raw, s_raw, _ = svds(W_U_T, k=k)
            s = np.abs(np.asarray(s_raw, dtype=np.float64).ravel())
            sort_idx = np.argsort(-s)
            s = s[sort_idx]
            U_wut = np.asarray(U_raw, dtype=np.float64)[:, sort_idx]
            del U_raw, s_raw
            
            PR = float((np.sum(s) ** 2) / np.sum(s ** 2))
            print(f"    PR={PR:.1f}, PR/d={PR/d_model:.4f}")
            
            # 测量每个属性在每个层的ratio
            test_attrs = list(attr_directions.keys())[:4]  # 取4个属性
            sample_layers = get_sample_layers(n_layers, n_samples=6)
            
            for attr in test_attrs:
                if attr not in attr_directions:
                    continue
                direction = attr_directions[attr]
                
                if attr not in all_ratio_curves:
                    all_ratio_curves[attr] = {}
                
                inputs_base, inputs_interv, _, pos_ids = inject_at_embed(
                    model, tokenizer, device, PROMPT, direction, BETA
                )
                
                base_out = collect_layer_outputs(model, inputs_base, pos_ids, n_layers)
                interv_out = collect_layer_outputs(model, inputs_interv, pos_ids, n_layers)
                
                for li in sample_layers:
                    key = f"L{li}"
                    if key not in base_out or key not in interv_out:
                        continue
                    
                    if li not in all_ratio_curves[attr]:
                        all_ratio_curves[attr][li] = {}
                    
                    h_base = base_out[key][0, -1, :].numpy()
                    h_interv = interv_out[key][0, -1, :].numpy()
                    delta_h = h_interv - h_base
                    
                    # 用当前k的U_wut测量ratio
                    proj_coeffs = U_wut.T @ delta_h
                    proj_energy = np.sum(proj_coeffs ** 2)
                    delta_norm_sq = np.sum(delta_h ** 2)
                    ratio = min(proj_energy / max(delta_norm_sq, 1e-20), 1.0)
                    
                    all_ratio_curves[attr][li][k] = {
                        "ratio": float(ratio),
                        "delta_norm": float(np.sqrt(delta_norm_sq)),
                        "PR": PR,
                    }
            
            del U_wut, s
            gc.collect()
            
        except Exception as e:
            print(f"    k={k} failed: {e}")
    
    del W_U_T; gc.collect()
    
    # Step 2: 汇总ratio(k)曲线
    print(f"\n  Step 2: ratio(k)曲线汇总...")
    
    # 对所有属性和层取平均
    ratio_vs_k = {}
    for k in k_values:
        ratios_at_k = []
        for attr in all_ratio_curves:
            for li in all_ratio_curves[attr]:
                if k in all_ratio_curves[attr][li]:
                    ratios_at_k.append(all_ratio_curves[attr][li][k]["ratio"])
        if ratios_at_k:
            ratio_vs_k[k] = {
                "mean_ratio": float(np.mean(ratios_at_k)),
                "std_ratio": float(np.std(ratios_at_k)),
                "n_samples": len(ratios_at_k),
            }
            print(f"    k={k}: ratio={np.mean(ratios_at_k):.4f}+/-{np.std(ratios_at_k):.4f}")
    
    # Step 3: 拟合ratio(k)曲线
    print(f"\n  Step 3: 拟合ratio(k)曲线...")
    
    if len(ratio_vs_k) >= 3:
        ks = np.array(sorted(ratio_vs_k.keys()), dtype=np.float64)
        ratios = np.array([ratio_vs_k[k]["mean_ratio"] for k in ks])
        
        # 模型a: 指数饱和 ratio(k) = 1 - a*exp(-b*k)
        try:
            from scipy.optimize import curve_fit
            def exp_model(k, a, b):
                return 1.0 - a * np.exp(-b * k)
            popt_exp, _ = curve_fit(exp_model, ks, ratios, p0=[1.0, 0.001], maxfev=10000)
            r_exp = exp_model(ks, *popt_exp)
            residual_exp = np.sum((ratios - r_exp) ** 2)
            print(f"    指数饱和: ratio(k)=1-{popt_exp[0]:.4f}*exp(-{popt_exp[1]:.6f}*k), residual={residual_exp:.6f}")
        except Exception as e:
            popt_exp = None
            residual_exp = float('inf')
            print(f"    指数饱和拟合失败: {e}")
        
        # 模型b: 幂律 ratio(k) = a * (k/d)^b
        try:
            def power_model(k, a, b):
                return a * (k / d_model) ** b
            popt_pow, _ = curve_fit(power_model, ks, ratios, p0=[1.0, 0.5], maxfev=10000)
            r_pow = power_model(ks, *popt_pow)
            residual_pow = np.sum((ratios - r_pow) ** 2)
            print(f"    幂律: ratio(k)={popt_pow[0]:.4f}*(k/d)^{popt_pow[1]:.4f}, residual={residual_pow:.6f}")
        except Exception as e:
            popt_pow = None
            residual_pow = float('inf')
            print(f"    幂律拟合失败: {e}")
        
        # 模型c: 线性 ratio(k) = a * k/d
        try:
            def linear_model(k, a):
                return a * (k / d_model)
            popt_lin, _ = curve_fit(linear_model, ks, ratios, p0=[1.0])
            r_lin = linear_model(ks, *popt_lin)
            residual_lin = np.sum((ratios - r_lin) ** 2)
            print(f"    线性: ratio(k)={popt_lin[0]:.4f}*(k/d), residual={residual_lin:.6f}")
        except Exception as e:
            popt_lin = None
            residual_lin = float('inf')
            print(f"    线性拟合失败: {e}")
        
        # 选择最佳模型
        models = []
        if popt_exp is not None:
            models.append(("exponential", residual_exp, popt_exp))
        if popt_pow is not None:
            models.append(("power", residual_pow, popt_pow))
        if popt_lin is not None:
            models.append(("linear", residual_lin, popt_lin))
        
        if models:
            best = min(models, key=lambda x: x[1])
            print(f"\n    最佳模型: {best[0]} (residual={best[1]:.6f})")
            results["best_model"] = best[0]
            results["model_params"] = {
                "exponential": popt_exp.tolist() if popt_exp is not None else None,
                "power": popt_pow.tolist() if popt_pow is not None else None,
                "linear": popt_lin.tolist() if popt_lin is not None else None,
            }
            results["model_residuals"] = {
                "exponential": float(residual_exp) if popt_exp is not None else None,
                "power": float(residual_pow) if popt_pow is not None else None,
                "linear": float(residual_lin) if popt_lin is not None else None,
            }
    
    results["ratio_vs_k"] = ratio_vs_k
    results["all_ratio_curves"] = {
        attr: {str(li): {str(k): v for k, v in layers.items()} 
               for li, layers in attr_data.items()}
        for attr, attr_data in all_ratio_curves.items()
    }
    
    return results


# ================================================================
# P453: 逃逸方向功能识别
# ================================================================

def run_p453(model, tokenizer, device, model_info):
    """
    P453: 逃逸方向功能识别
    
    原理:
      P450发现逃逸能量集中在6-7个方向(PR=5.9-7.4)。
      这些逃逸方向与W_U完全正交(对齐度=0.000)。
      它们对应什么功能?
      
      候选方向:
      1. LayerNorm方向: LN的gamma参数定义了"正则化方向"
      2. 注意力Q/K/V方向: 注意力计算可能沿特定方向
      3. MLP up/down方向: MLP的计算可能沿特定方向
      4. 残差流方向: 残差连接保持的方向
      
    方法:
      1. 收集逃逸向量，做SVD获取逃逸方向(前7个)
      2. 计算逃逸方向与各种"功能方向"的余弦相似度
      3. 找出最对齐的功能方向
    """
    print(f"\n{'='*60}")
    print(f"P453: 逃逸方向功能识别 - {model_info.name}")
    print(f"{'='*60}")
    
    results = {}
    d_model = model_info.d_model
    n_layers = model_info.n_layers
    layers = get_layers(model)
    
    # Step 1: 收集逃逸向量
    W_U = get_W_U(model)
    
    attr_directions = {}
    for attr in TEST_ATTRS:
        direction, _ = get_attr_direction(model, tokenizer, attr, W_U=W_U)
        if direction is not None:
            attr_directions[attr] = direction
    
    # W_U SVD (k=300用于定义"行空间")
    W_U_T = W_U.T.astype(np.float32)
    del W_U; import gc; gc.collect()
    
    max_k = min(W_U_T.shape[0], W_U_T.shape[1]) - 2
    k_svd = min(300, max_k)
    U_raw, s_raw, _ = svds(W_U_T, k=k_svd)
    del W_U_T; gc.collect()
    
    s = np.abs(np.asarray(s_raw, dtype=np.float64).ravel())
    sort_idx = np.argsort(-s)
    s = s[sort_idx]
    U_wut = np.asarray(U_raw, dtype=np.float64)[:, sort_idx]
    del U_raw, s_raw; gc.collect()
    
    print(f"  W_U SVD: k={k_svd}, PR={float((np.sum(s)**2)/np.sum(s**2)):.1f}")
    
    # 收集逃逸向量
    escape_vectors = []
    
    for attr in list(attr_directions.keys())[:4]:
        direction = attr_directions[attr]
        
        inputs_base, inputs_interv, _, pos_ids = inject_at_embed(
            model, tokenizer, device, PROMPT, direction, BETA
        )
        
        base_out = collect_layer_outputs(model, inputs_base, pos_ids, n_layers)
        interv_out = collect_layer_outputs(model, inputs_interv, pos_ids, n_layers)
        
        for li in get_sample_layers(n_layers, n_samples=6):
            key = f"L{li}"
            if key not in base_out or key not in interv_out:
                continue
            
            h_base = base_out[key][0, -1, :].numpy()
            h_interv = interv_out[key][0, -1, :].numpy()
            delta_h = h_interv - h_base
            
            # 逃逸分量
            proj_coeffs = U_wut.T @ delta_h
            proj = U_wut @ proj_coeffs
            escape = delta_h - proj
            
            escape_vectors.append({
                "attr": attr,
                "layer": li,
                "escape": escape,
            })
    
    print(f"  收集了 {len(escape_vectors)} 个逃逸向量")
    
    if len(escape_vectors) < 3:
        results["error"] = "Too few escape vectors"
        return results
    
    # Step 2: 逃逸方向的SVD
    escape_matrix = np.column_stack([e["escape"] for e in escape_vectors])
    U_esc, s_esc, Vt_esc = np.linalg.svd(escape_matrix, full_matrices=False)
    
    # 取前10个逃逸方向
    n_escape_dirs = min(10, len(s_esc))
    escape_dirs = U_esc[:, :n_escape_dirs]  # [d_model, n_escape_dirs]
    
    print(f"  逃逸SVD: top {n_escape_dirs} 奇异值 = {s_esc[:n_escape_dirs].tolist()}")
    
    # Step 3: 计算逃逸方向与功能方向的对齐度
    print(f"\n  Step 3: 逃逸方向功能对齐度...")
    
    alignment_results = {}
    
    # 3a. 与LayerNorm gamma的对齐度
    print(f"  3a. LayerNorm gamma对齐度...")
    ln_alignments = []
    for li in [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]:
        if li >= n_layers:
            continue
        lw = get_layer_weights(layers[li], d_model, model_info.mlp_type)
        
        if lw.input_layernorm_weight is not None:
            ln_gamma = np.asarray(lw.input_layernorm_weight)
            ln_gamma_norm = ln_gamma / max(np.linalg.norm(ln_gamma), 1e-10)
            
            for di in range(n_escape_dirs):
                esc_dir = escape_dirs[:, di]
                esc_dir_norm = esc_dir / max(np.linalg.norm(esc_dir), 1e-10)
                cos_val = float(np.abs(np.dot(ln_gamma_norm, esc_dir_norm)))
                ln_alignments.append({
                    "layer": li,
                    "escape_dir": di,
                    "cos": cos_val,
                })
    
    if ln_alignments:
        mean_ln_cos = np.mean([a["cos"] for a in ln_alignments])
        max_ln_cos = np.max([a["cos"] for a in ln_alignments])
        print(f"    LN gamma vs 逃逸方向: mean|cos|={mean_ln_cos:.4f}, max|cos|={max_ln_cos:.4f}")
        alignment_results["layernorm_gamma"] = {
            "mean_abs_cos": float(mean_ln_cos),
            "max_abs_cos": float(max_ln_cos),
            "details": ln_alignments[:20],  # 只保存前20个
        }
    
    # 3b. 与注意力Q/K/V方向的对齐度
    print(f"  3b. 注意力Q/K/V对齐度...")
    attn_alignments = []
    for li in [0, n_layers//2, n_layers-1]:
        if li >= n_layers:
            continue
        lw = get_layer_weights(layers[li], d_model, model_info.mlp_type)
        
        # W_q: [n_heads*d_head, d_model] → 取前几个奇异向量
        if lw.W_q is not None:
            W_q = np.asarray(lw.W_q).astype(np.float32)
            try:
                k_attn = min(10, min(W_q.shape) - 2)
                U_q, s_q, _ = svds(W_q.T, k=k_attn)  # U_q: [d_model, k_attn]
                U_q = np.asarray(U_q, dtype=np.float64)
                
                for di in range(min(5, n_escape_dirs)):
                    esc_dir = escape_dirs[:, di]
                    # 计算与Q的top-k奇异向量的最大对齐度
                    cos_vals = np.abs(U_q.T @ esc_dir)
                    cos_vals = cos_vals / max(np.linalg.norm(esc_dir), 1e-10)
                    max_cos = float(np.max(cos_vals))
                    attn_alignments.append({
                        "layer": li,
                        "escape_dir": di,
                        "component": "W_q",
                        "max_cos": max_cos,
                    })
                del U_q
            except:
                pass
        
        # W_k
        if lw.W_k is not None:
            W_k = np.asarray(lw.W_k).astype(np.float32)
            try:
                k_attn = min(10, min(W_k.shape) - 2)
                U_k, s_k, _ = svds(W_k.T, k=k_attn)
                U_k = np.asarray(U_k, dtype=np.float64)
                
                for di in range(min(5, n_escape_dirs)):
                    esc_dir = escape_dirs[:, di]
                    cos_vals = np.abs(U_k.T @ esc_dir)
                    cos_vals = cos_vals / max(np.linalg.norm(esc_dir), 1e-10)
                    max_cos = float(np.max(cos_vals))
                    attn_alignments.append({
                        "layer": li,
                        "escape_dir": di,
                        "component": "W_k",
                        "max_cos": max_cos,
                    })
                del U_k
            except:
                pass
    
    if attn_alignments:
        q_aligns = [a["max_cos"] for a in attn_alignments if a["component"] == "W_q"]
        k_aligns = [a["max_cos"] for a in attn_alignments if a["component"] == "W_k"]
        
        if q_aligns:
            print(f"    W_q vs 逃逸方向: mean max|cos|={np.mean(q_aligns):.4f}")
        if k_aligns:
            print(f"    W_k vs 逃逸方向: mean max|cos|={np.mean(k_aligns):.4f}")
        
        alignment_results["attention"] = {
            "W_q_mean_max_cos": float(np.mean(q_aligns)) if q_aligns else None,
            "W_k_mean_max_cos": float(np.mean(k_aligns)) if k_aligns else None,
            "details": attn_alignments[:20],
        }
    
    # 3c. 与MLP up/down方向的对齐度
    print(f"  3c. MLP W_up/W_down对齐度...")
    mlp_alignments = []
    for li in [0, n_layers//2, n_layers-1]:
        if li >= n_layers:
            continue
        lw = get_layer_weights(layers[li], d_model, model_info.mlp_type)
        
        # W_up: [d_mlp, d_model]
        if lw.W_up is not None:
            W_up = np.asarray(lw.W_up).astype(np.float32)
            try:
                k_mlp = min(10, min(W_up.shape) - 2)
                U_up, s_up, _ = svds(W_up.T, k=k_mlp)
                U_up = np.asarray(U_up, dtype=np.float64)
                
                for di in range(min(5, n_escape_dirs)):
                    esc_dir = escape_dirs[:, di]
                    cos_vals = np.abs(U_up.T @ esc_dir)
                    cos_vals = cos_vals / max(np.linalg.norm(esc_dir), 1e-10)
                    max_cos = float(np.max(cos_vals))
                    mlp_alignments.append({
                        "layer": li,
                        "escape_dir": di,
                        "component": "W_up",
                        "max_cos": max_cos,
                    })
                del U_up
            except:
                pass
        
        # W_down: [d_model, d_mlp]
        if lw.W_down is not None:
            W_down = np.asarray(lw.W_down).astype(np.float32)
            try:
                k_mlp = min(10, min(W_down.shape) - 2)
                U_down, s_down, _ = svds(W_down, k=k_mlp)
                U_down = np.asarray(U_down, dtype=np.float64)
                
                for di in range(min(5, n_escape_dirs)):
                    esc_dir = escape_dirs[:, di]
                    cos_vals = np.abs(U_down.T @ esc_dir)
                    cos_vals = cos_vals / max(np.linalg.norm(esc_dir), 1e-10)
                    max_cos = float(np.max(cos_vals))
                    mlp_alignments.append({
                        "layer": li,
                        "escape_dir": di,
                        "component": "W_down",
                        "max_cos": max_cos,
                    })
                del U_down
            except:
                pass
    
    if mlp_alignments:
        up_aligns = [a["max_cos"] for a in mlp_alignments if a["component"] == "W_up"]
        down_aligns = [a["max_cos"] for a in mlp_alignments if a["component"] == "W_down"]
        
        if up_aligns:
            print(f"    W_up vs 逃逸方向: mean max|cos|={np.mean(up_aligns):.4f}")
        if down_aligns:
            print(f"    W_down vs 逃逸方向: mean max|cos|={np.mean(down_aligns):.4f}")
        
        alignment_results["mlp"] = {
            "W_up_mean_max_cos": float(np.mean(up_aligns)) if up_aligns else None,
            "W_down_mean_max_cos": float(np.mean(down_aligns)) if down_aligns else None,
            "details": mlp_alignments[:20],
        }
    
    # 3d. 与残差流方向的对齐度
    # 残差流方向 = 信号自身的方向（属性注入方向在传播后的变化）
    print(f"  3d. 残差流(注入方向)对齐度...")
    residual_alignments = []
    for attr in list(attr_directions.keys())[:4]:
        direction = attr_directions[attr]
        dir_norm = direction / max(np.linalg.norm(direction), 1e-10)
        
        for di in range(min(5, n_escape_dirs)):
            esc_dir = escape_dirs[:, di]
            esc_dir_norm = esc_dir / max(np.linalg.norm(esc_dir), 1e-10)
            cos_val = float(np.abs(np.dot(dir_norm, esc_dir_norm)))
            residual_alignments.append({
                "attr": attr,
                "escape_dir": di,
                "cos": cos_val,
            })
    
    if residual_alignments:
        mean_res_cos = np.mean([a["cos"] for a in residual_alignments])
        print(f"    注入方向 vs 逃逸方向: mean|cos|={mean_res_cos:.4f}")
        alignment_results["residual_inject"] = {
            "mean_abs_cos": float(mean_res_cos),
        }
    
    # 3e. 与随机方向的对齐度(基线)
    print(f"  3e. 随机方向对齐度(基线)...")
    np.random.seed(42)
    n_random = 100
    random_cos_vals = []
    for _ in range(n_random):
        rand_dir = np.random.randn(d_model)
        rand_dir = rand_dir / np.linalg.norm(rand_dir)
        for di in range(min(5, n_escape_dirs)):
            esc_dir = escape_dirs[:, di]
            esc_dir_norm = esc_dir / max(np.linalg.norm(esc_dir), 1e-10)
            cos_val = float(np.abs(np.dot(rand_dir, esc_dir_norm)))
            random_cos_vals.append(cos_val)
    
    mean_random_cos = np.mean(random_cos_vals)
    print(f"    随机方向 vs 逃逸方向: mean|cos|={mean_random_cos:.4f} (基线)")
    alignment_results["random_baseline"] = {"mean_abs_cos": float(mean_random_cos)}
    
    # 汇总: 哪个功能方向最对齐?
    print(f"\n  对齐度汇总:")
    print(f"    随机基线: {mean_random_cos:.4f}")
    if "layernorm_gamma" in alignment_results:
        print(f"    LN gamma: {alignment_results['layernorm_gamma']['mean_abs_cos']:.4f}")
    if "attention" in alignment_results:
        if alignment_results["attention"]["W_q_mean_max_cos"]:
            print(f"    W_q: {alignment_results['attention']['W_q_mean_max_cos']:.4f}")
        if alignment_results["attention"]["W_k_mean_max_cos"]:
            print(f"    W_k: {alignment_results['attention']['W_k_mean_max_cos']:.4f}")
    if "mlp" in alignment_results:
        if alignment_results["mlp"]["W_up_mean_max_cos"]:
            print(f"    W_up: {alignment_results['mlp']['W_up_mean_max_cos']:.4f}")
        if alignment_results["mlp"]["W_down_mean_max_cos"]:
            print(f"    W_down: {alignment_results['mlp']['W_down_mean_max_cos']:.4f}")
    if "residual_inject" in alignment_results:
        print(f"    注入方向: {alignment_results['residual_inject']['mean_abs_cos']:.4f}")
    
    results["escape_dirs_singular_values"] = s_esc[:n_escape_dirs].tolist()
    results["alignment"] = alignment_results
    
    # 清理
    for e in escape_vectors:
        del e["escape"]
    
    return results


# ================================================================
# P454: 频谱耦合函数拟合
# ================================================================

def run_p454(model, tokenizer, device, model_info):
    """
    P454: 频谱耦合函数拟合
    
    原理:
      假设ratio(k)的形状由信号在W_U不同奇异值方向上的投影能量分布决定。
      
      如果信号在W_U的第i个奇异向量上的投影能量为e_i，则:
      ratio(k) = (e_1 + e_2 + ... + e_k) / (e_1 + ... + e_d)
      
      如果e_i ∝ s_i^alpha (与奇异值成正比)，则不同的alpha给出不同的ratio(k)曲线。
      
    方法:
      1. 用k=max的SVD获取完整奇异值谱和U矩阵
      2. 对真实信号，计算每个奇异值方向上的投影能量e_i
      3. 分析e_i与s_i的关系
      4. 拟合最优alpha
    """
    print(f"\n{'='*60}")
    print(f"P454: 频谱耦合函数拟合 - {model_info.name}")
    print(f"{'='*60}")
    
    results = {}
    d_model = model_info.d_model
    n_layers = model_info.n_layers
    
    # W_U SVD (大k)
    W_U = get_W_U(model)
    direction, _ = get_attr_direction(model, tokenizer, "red", W_U=W_U)
    
    W_U_T = W_U.T.astype(np.float32)
    del W_U; import gc; gc.collect()
    
    max_k = min(W_U_T.shape[0], W_U_T.shape[1]) - 2
    if model_info.name == "glm4":
        max_k = min(max_k, 1500)
    k_svd = max_k
    
    print(f"  对W_U^T做SVD, k={k_svd}...")
    U_raw, s_raw, _ = svds(W_U_T, k=k_svd)
    del W_U_T; gc.collect()
    
    s = np.abs(np.asarray(s_raw, dtype=np.float64).ravel())
    sort_idx = np.argsort(-s)
    s = s[sort_idx]
    U_wut = np.asarray(U_raw, dtype=np.float64)[:, sort_idx]
    del U_raw, s_raw; gc.collect()
    
    print(f"  SVD完成, top5 SVs={s[:5].tolist()}, tail5={s[-5:].tolist()}")
    
    if direction is None:
        results["error"] = "No direction available"
        return results
    
    # 对多个属性×层，计算投影能量频谱
    print(f"  计算投影能量频谱...")
    
    attr_directions_all = {}
    W_U2 = get_W_U(model)
    for attr in TEST_ATTRS:
        d, _ = get_attr_direction(model, tokenizer, attr, W_U=W_U2)
        if d is not None:
            attr_directions_all[attr] = d
    del W_U2; gc.collect()
    
    sample_layers = get_sample_layers(n_layers, n_samples=4)
    
    all_spectra = []  # 每个元素的e_i值
    all_s_values = s  # 共享的奇异值谱
    
    for attr in list(attr_directions_all.keys())[:3]:
        dir_attr = attr_directions_all[attr]
        
        inputs_base, inputs_interv, _, pos_ids = inject_at_embed(
            model, tokenizer, device, PROMPT, dir_attr, BETA
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
            
            # 投影系数
            proj_coeffs = U_wut.T @ delta_h  # [k_svd]
            proj_energy_per_sv = proj_coeffs ** 2  # 每个方向的能量
            
            all_spectra.append({
                "attr": attr,
                "layer": li,
                "energy_per_sv": proj_energy_per_sv,
                "total_proj_energy": float(np.sum(proj_energy_per_sv)),
                "delta_norm": float(np.linalg.norm(delta_h)),
            })
    
    print(f"  收集了 {len(all_spectra)} 个投影频谱")
    
    if len(all_spectra) < 3:
        results["error"] = "Too few spectra"
        return results
    
    # 分析: e_i vs s_i 的关系
    print(f"\n  分析 e_i vs s_i 关系...")
    
    # 平均频谱
    mean_energy = np.zeros(k_svd)
    for spec in all_spectra:
        mean_energy += spec["energy_per_sv"]
    mean_energy /= len(all_spectra)
    
    # 排除零值
    valid = (mean_energy > 0) & (s > 0)
    
    # 对数空间拟合: log(e_i) = alpha * log(s_i) + beta
    if np.sum(valid) > 10:
        log_s = np.log10(s[valid])
        log_e = np.log10(mean_energy[valid])
        
        # 线性拟合
        from numpy.polynomial import polynomial as P
        coeffs = np.polyfit(log_s, log_e, 1)
        alpha_fit = coeffs[0]
        beta_fit = coeffs[1]
        
        # 相关性
        corr_log = np.corrcoef(log_s, log_e)[0, 1]
        
        print(f"    log10(e_i) = {alpha_fit:.3f} * log10(s_i) + {beta_fit:.3f}")
        print(f"    相关性 r={corr_log:.4f}")
        print(f"    alpha={alpha_fit:.3f}: ", end="")
        if alpha_fit > 0.5:
            print("投影能量集中在大奇异值方向(聚焦)")
        elif alpha_fit > -0.5:
            print("投影能量与奇异值无关(均匀)")
        else:
            print("投影能量集中在小奇异值方向(反聚焦)")
        
        # 累积能量曲线
        cum_energy = np.cumsum(mean_energy) / np.sum(mean_energy)
        k_50 = int(np.searchsorted(cum_energy, 0.50)) + 1
        k_90 = int(np.searchsorted(cum_energy, 0.90)) + 1
        k_95 = int(np.searchsorted(cum_energy, 0.95)) + 1
        
        print(f"\n    信号投影能量累积:")
        print(f"      50%能量在前{k_50}个方向 (占{k_50/k_svd*100:.1f}%)")
        print(f"      90%能量在前{k_90}个方向 (占{k_90/k_svd*100:.1f}%)")
        print(f"      95%能量在前{k_95}个方向 (占{k_95/k_svd*100:.1f}%)")
        
        # 对比: 如果e_i ∝ s_i^alpha, 预测的ratio(k)是什么?
        # 预测ratio(k) = Σ(s_i^alpha * s_i^2)[:k] / Σ(s_i^alpha * s_i^2)[:]
        # (因为e_i ∝ s_i^alpha, 每个方向的能量权重是s_i^alpha)
        # 但更精确: ratio(k) = Σe_i[:k] / Σe_i[:]
        # 用实测e_i直接计算
        predicted_ratio_vs_k = cum_energy
        
        # 用拟合模型预测
        if abs(alpha_fit) > 0.01:
            weights_pred = s ** alpha_fit
            cum_pred = np.cumsum(weights_pred) / np.sum(weights_pred)
        else:
            cum_pred = np.arange(1, k_svd+1) / k_svd  # 均匀
        
        results["spectral_coupling"] = {
            "alpha": float(alpha_fit),
            "beta": float(beta_fit),
            "r_log": float(corr_log),
            "k_50": k_50,
            "k_90": k_90,
            "k_95": k_95,
            "cum_energy_sample": cum_energy[::max(1, k_svd//50)].tolist(),
            "cum_pred_sample": cum_pred[::max(1, k_svd//50)].tolist(),
        }
    
    # 对不同alpha值的ratio(k)预测
    print(f"\n  频谱耦合模型对比...")
    for alpha_test in [-2, -1, 0, 0.5, 1, 2]:
        weights = s ** alpha_test
        cum_w = np.cumsum(weights) / np.sum(weights)
        # 在k=300处的预测ratio
        ratio_at_300 = float(cum_w[min(299, len(cum_w)-1)])
        print(f"    alpha={alpha_test:5.1f}: ratio(k=300)={ratio_at_300:.4f}")
    
    # 清理
    for spec in all_spectra:
        del spec["energy_per_sv"]
    
    results["n_spectra"] = len(all_spectra)
    results["k_svd"] = k_svd
    
    return results


# ================================================================
# P455: 信号频谱与W_U频谱的统计关系
# ================================================================

def run_p455(model, tokenizer, device, model_info):
    """
    P455: 信号频谱与W_U频谱的统计关系
    
    原理:
      核心问题: 信号的投影能量e_i与W_U奇异值s_i的关系是什么?
      
      如果e_i ∝ s_i^0 (均匀) → ratio(k) = k/d → 随机
      如果e_i ∝ s_i^1 → 信号集中在大SV方向 → ratio(k)快速饱和
      如果e_i ∝ s_i^-1 → 信号集中在小SV方向 → ratio(k)缓慢增长
      
      更深入: e_i 的分布是否可以分解为"随机基线+训练结构"?
      
    方法:
      1. 对真实信号计算e_i
      2. 对随机信号(随机正交矩阵旋转)计算e_i
      3. 对比: 真实e_i vs 随机e_i
      4. 训练结构 = 真实e_i - 随机e_i
    """
    print(f"\n{'='*60}")
    print(f"P455: 信号频谱与W_U频谱统计关系 - {model_info.name}")
    print(f"{'='*60}")
    
    results = {}
    d_model = model_info.d_model
    n_layers = model_info.n_layers
    
    # W_U SVD
    W_U = get_W_U(model)
    direction, _ = get_attr_direction(model, tokenizer, "red", W_U=W_U)
    
    W_U_T = W_U.T.astype(np.float32)
    del W_U; import gc; gc.collect()
    
    max_k = min(W_U_T.shape[0], W_U_T.shape[1]) - 2
    if model_info.name == "glm4":
        max_k = min(max_k, 1500)
    k_svd = max_k
    
    print(f"  SVD k={k_svd}...")
    U_raw, s_raw, _ = svds(W_U_T, k=k_svd)
    del W_U_T; gc.collect()
    
    s = np.abs(np.asarray(s_raw, dtype=np.float64).ravel())
    sort_idx = np.argsort(-s)
    s = s[sort_idx]
    U_wut = np.asarray(U_raw, dtype=np.float64)[:, sort_idx]
    del U_raw, s_raw; gc.collect()
    
    if direction is None:
        results["error"] = "No direction"
        return results
    
    # Step 1: 真实信号的频谱
    print(f"\n  Step 1: 真实信号频谱...")
    
    inputs_base, inputs_interv, _, pos_ids = inject_at_embed(
        model, tokenizer, device, PROMPT, direction, BETA
    )
    
    base_out = collect_layer_outputs(model, inputs_base, pos_ids, n_layers)
    interv_out = collect_layer_outputs(model, inputs_interv, pos_ids, n_layers)
    
    # 取中间层
    mid_layer = n_layers // 2
    key = f"L{mid_layer}"
    if key in base_out and key in interv_out:
        h_base = base_out[key][0, -1, :].numpy()
        h_interv = interv_out[key][0, -1, :].numpy()
        delta_real = h_interv - h_base
    else:
        delta_real = BETA * direction
    
    # 真实信号的投影能量频谱
    proj_real = U_wut.T @ delta_real  # [k_svd]
    e_real = proj_real ** 2
    
    ratio_real = np.sum(e_real) / max(np.sum(delta_real ** 2), 1e-20)
    print(f"    真实ratio={ratio_real:.4f}")
    
    # Step 2: 随机信号的频谱
    print(f"\n  Step 2: 随机信号频谱 (N=20)...")
    
    N_RANDOM = 20
    e_random_list = []
    np.random.seed(42)
    
    for i in range(N_RANDOM):
        # 随机正交旋转
        G = np.random.randn(d_model, d_model).astype(np.float32)
        Q, _ = np.linalg.qr(G)
        delta_rotated = Q @ delta_real
        
        proj_rand = U_wut.T @ delta_rotated
        e_rand = proj_rand ** 2
        e_random_list.append(e_rand)
    
    e_random_mean = np.mean(e_random_list, axis=0)
    e_random_std = np.std(e_random_list, axis=0)
    
    ratio_random_mean = np.mean([np.sum(e) / max(np.sum(delta_real ** 2), 1e-20) for e in e_random_list])
    print(f"    随机ratio={ratio_random_mean:.4f}")
    
    # Step 3: 对比真实vs随机
    print(f"\n  Step 3: 真实 vs 随机频谱对比...")
    
    # 频谱偏差 = 真实e_i / 随机e_i
    valid = e_random_mean > 1e-20
    spectral_bias = np.ones(k_svd)
    spectral_bias[valid] = e_real[valid] / e_random_mean[valid]
    
    # 分区统计: 前10%, 中间, 后10%
    n_10pct = max(1, k_svd // 10)
    
    bias_top10 = np.mean(spectral_bias[:n_10pct])
    bias_mid = np.mean(spectral_bias[n_10pct:-n_10pct])
    bias_bot10 = np.mean(spectral_bias[-n_10pct:])
    
    print(f"    频谱偏差(真实/随机):")
    print(f"      前10%(大SV): {bias_top10:.2f}x")
    print(f"      中间80%:     {bias_mid:.2f}x")
    print(f"      后10%(小SV): {bias_bot10:.2f}x")
    
    # 相关分析: spectral_bias与s_i的关系
    corr_bias_sv = np.corrcoef(spectral_bias[valid], np.log10(s[valid]))[0, 1] if np.sum(valid) > 10 else 0
    print(f"    偏差与SV大小相关: r={corr_bias_sv:.4f}")
    
    # Step 4: 训练结构 = 真实 - 随机
    training_structure = e_real - e_random_mean
    training_frac = np.sum(np.maximum(training_structure, 0)) / max(np.sum(e_real), 1e-20)
    
    print(f"\n  Step 4: 训练结构分析...")
    print(f"    训练结构占比: {training_frac*100:.1f}%")
    
    # 训练结构集中在哪些方向?
    # 找出training_structure最大的方向
    top_struct_idx = np.argsort(-training_structure)[:10]
    print(f"    训练结构最大的10个方向(SV排名): {top_struct_idx.tolist()}")
    print(f"    对应SV值: {s[top_struct_idx].tolist()}")
    print(f"    训练结构值: {training_structure[top_struct_idx].tolist()}")
    
    # 负训练结构(训练抑制的方向)
    bot_struct_idx = np.argsort(training_structure)[:10]
    print(f"    训练抑制最大的10个方向(SV排名): {bot_struct_idx.tolist()}")
    
    results["real_ratio"] = float(ratio_real)
    results["random_ratio_mean"] = float(ratio_random_mean)
    results["spectral_bias"] = {
        "top10pct": float(bias_top10),
        "mid80pct": float(bias_mid),
        "bot10pct": float(bias_bot10),
        "corr_with_sv": float(corr_bias_sv),
    }
    results["training_structure"] = {
        "fraction": float(training_frac),
        "top10_indices": top_struct_idx.tolist(),
        "top10_sv_values": s[top_struct_idx].tolist(),
        "top10_values": training_structure[top_struct_idx].tolist(),
        "bot10_indices": bot_struct_idx.tolist(),
    }
    results["spectral_bias_sample"] = spectral_bias[::max(1, k_svd//50)].tolist()
    
    return results


# ================================================================
# 主流程
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
        "p452": run_p452,
        "p453": run_p453,
        "p454": run_p454,
        "p455": run_p455,
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
        out_file = OUT_DIR / f"phase_xciii_p452_455_{model_name}_{timestamp}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nResults saved to {out_file}")
        
        release_model(model)


def main():
    parser = argparse.ArgumentParser(description=f"{PHASE_NAME}: {EXPERIMENT_DESC}")
    parser.add_argument("--model", type=str, required=True,
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--experiment", type=str, default="all",
                       choices=["all", "p452", "p453", "p454", "p455"])
    args = parser.parse_args()
    
    experiments = [args.experiment] if args.experiment != "all" else None
    run_single_model(args.model, experiments)


if __name__ == "__main__":
    main()
