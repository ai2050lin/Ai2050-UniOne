"""
Phase LXXVI-P387/388/389: 正交性退化机制 + 交互项精确公式 + W_up演化动力学
=========================================================================

阶段A核心任务:

P387: 正交性退化 = 混合还是旋转？★★★最关键★★★
  - P384发现: 中间层正交性被系统性破坏, 但只测了cos
  - 无法区分: 维度方向"混合"(两个方向变成同一个) vs "旋转"(方向保持独立但旋转到新位置)
  - 方法: 对每层的Δh向量做PCA分解, 分析主成分方向
    - 如果混合: PCA维度降低, 三个Δh投影到同几个主成分
    - 如果旋转: PCA维度保持3, 主成分与原始W_lm方向有旋转关系
  - 进一步: 计算旋转矩阵R, 使得Δh_mid ≈ R × Δh_L0
    - 如果R是正交矩阵 → 纯旋转
    - 如果R是奇异矩阵 → 混合坍缩
  - 关键指标: R的有效秩, R的奇异值分布

P388: 交互项精确公式
  - P385发现: 交互项≈3.4(β=5), 随β先增后减
  - 目标: 确定 f(β, orthogonality, W_up) 的数学形式
  - 方法: 
    1. 精细β扫描(0.5-20, 步长0.5) 测量交互项
    2. 拟合候选公式:
       - 线性: interaction = a*β
       - 二次: interaction = a*β^2
       - S型: interaction = a*β/(1+b*β)  (有饱和)
       - 指数: interaction = a*(1-exp(-b*β))
       - 高斯: interaction = a*β*exp(-b*β^2) (先增后减)
    3. 交叉验证: 用50%数据拟合, 50%验证

P389: W_up训练动力学推断
  - 虽然无法观察训练过程, 但可以从模型的层间W_up分布推断训练动力学
  - 方法:
    1. 绘制每层W_up谱范数, W_gate谱范数, W_down谱范数
    2. 计算W_up/W_gate/W_down的层间相关性
    3. 分析W_up与attention head维度的关系
    4. 推断: W_up的大小是否由层的"任务"决定? (早期层需要小W_up保留信号)

实验模型: qwen3 -> glm4 -> deepseek7b (串行, 避免GPU OOM)
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

OUT_DIR = Path("tests/glm5_temp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 风格/逻辑/语法维度词对
STYLE_PAIRS = [("formal", "informal"), ("polite", "rude")]
LOGIC_PAIRS = [("true", "false"), ("correct", "wrong")]
GRAMMAR_PAIRS = [("active", "passive"), ("singular", "plural")]

PROMPT = "The apple is"

MODEL_CONFIGS = {
    "qwen3": {
        "path": r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c",
        "trust_remote_code": True,
        "use_fast": False,
    },
    "glm4": {
        "path": r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf",
        "trust_remote_code": True,
        "use_fast": False,
    },
    "deepseek7b": {
        "path": r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60",
        "trust_remote_code": True,
        "use_fast": False,
    },
}


def load_model(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    print(f"Loading {model_name}...")
    mdl = AutoModelForCausalLM.from_pretrained(
        cfg["path"], dtype=torch.bfloat16, trust_remote_code=cfg["trust_remote_code"],
        local_files_only=True, low_cpu_mem_usage=True, attn_implementation="eager", device_map="cpu",
    )
    if torch.cuda.is_available():
        mdl = mdl.to("cuda")
    mdl.eval()
    device = mdl.device
    tok = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=cfg["trust_remote_code"],
        local_files_only=True, use_fast=cfg["use_fast"],
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return mdl, tok, device


def get_w_lm_normed(model, tokenizer, word):
    tok_ids = tokenizer.encode(word, add_special_tokens=False)
    tok_id = tok_ids[0]
    lm_head = model.lm_head
    w = lm_head.weight[tok_id].detach().cpu().float()
    w_norm = w / w.norm()
    return w_norm.numpy(), w.numpy()


def get_dimension_direction(model, tokenizer, word_pos, word_neg):
    w_pos, _ = get_w_lm_normed(model, tokenizer, word_pos)
    w_neg, _ = get_w_lm_normed(model, tokenizer, word_neg)
    diff = w_pos - w_neg
    norm = np.linalg.norm(diff)
    if norm < 1e-8:
        return w_pos, 0.0
    return diff / norm, norm


def get_layers(model, max_layers=None):
    if hasattr(model.model, "layers"):
        all_layers = list(model.model.layers)
    elif hasattr(model.model, "encoder"):
        all_layers = list(model.model.encoder.layers)
    else:
        raise ValueError("Cannot find layers")
    if max_layers is None:
        return all_layers
    return all_layers[:max_layers]


def get_mlp(model, layer):
    if hasattr(layer, "mlp"):
        return layer.mlp
    elif hasattr(layer, "feed_forward"):
        return layer.feed_forward
    raise ValueError("Cannot find MLP")


# ========== P387: 正交性退化 = 混合还是旋转？★★★最关键★★★ ==========

def run_p387(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P387: Orthogonality degradation - Mixing vs Rotation - {model_name}")
    print(f"{'='*60}")

    embed = model.get_input_embeddings()
    dim_names = ["style", "logic", "grammar"]
    dim_pairs = [STYLE_PAIRS[0], LOGIC_PAIRS[0], GRAMMAR_PAIRS[0]]
    beta = 8.0

    n_layers_total = len(get_layers(model))
    scan_layers = [0, 1, 2, 3, 5, 8, 10, 15, 20]
    scan_layers = [l for l in scan_layers if l < n_layers_total]
    n_scan = max(scan_layers) + 1
    layers = get_layers(model, n_scan)

    # 计算三个维度的W_lm方向
    dim_directions = {}
    for name, (pos, neg) in zip(dim_names, dim_pairs):
        direction, norm = get_dimension_direction(model, tokenizer, pos, neg)
        dim_directions[name] = {"direction": direction, "norm": norm}

    toks = tokenizer(PROMPT, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    seq_len = input_ids.shape[1]
    inputs_embeds_base = embed(input_ids).detach().clone().to(model.dtype)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    # 捕获baseline每层hidden state
    captured_base = {}

    def make_hook(storage, key):
        def hook(module, input, output):
            if isinstance(output, tuple):
                storage[key] = output[0].detach().float()
            else:
                storage[key] = output.detach().float()
        return hook

    handles_base = []
    for i, layer in enumerate(layers):
        if i in scan_layers:
            handles_base.append(layer.register_forward_hook(make_hook(captured_base, f"L{i}")))

    with torch.no_grad():
        _ = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids)
    for h in handles_base:
        h.remove()

    # 对每个维度分别注入, 捕获每层hidden state
    delta_h_all = {}  # {dim_name: {layer: delta_h}}
    for dim_name, (pos, neg) in zip(dim_names, dim_pairs):
        direction = dim_directions[dim_name]["direction"]
        w_tensor = torch.tensor(direction, dtype=torch.float32, device=device)

        inputs_embeds_int = inputs_embeds_base.clone()
        inputs_embeds_int[0, -1, :] += (beta * w_tensor).to(model.dtype)

        captured_int = {}
        handles_int = []
        for i, layer in enumerate(layers):
            if i in scan_layers:
                handles_int.append(layer.register_forward_hook(make_hook(captured_int, f"L{i}")))

        with torch.no_grad():
            _ = model(inputs_embeds=inputs_embeds_int, position_ids=position_ids)
        for h in handles_int:
            h.remove()

        delta_h_all[dim_name] = {}
        for l in scan_layers:
            key = f"L{l}"
            if key in captured_int and key in captured_base:
                h_int = captured_int[key][0, -1, :].cpu().numpy()
                h_base = captured_base[key][0, -1, :].cpu().numpy()
                delta_h_all[dim_name][l] = h_int - h_base  # 保留未归一化的Δh

    # ===== 核心分析1: PCA维度分析 =====
    print(f"\n  === Analysis 1: PCA Dimensionality ===")
    pca_results = {}
    
    for l in scan_layers:
        # 堆叠三个维度的Δh
        dh_stack = np.stack([delta_h_all[n][l] for n in dim_names])  # [3, d_model]
        
        # SVD分解
        U, S, Vt = np.linalg.svd(dh_stack, full_matrices=False)
        
        # 前3个奇异值的比例
        total_energy = np.sum(S**2)
        energy_ratios = [(s**2 / total_energy) if total_energy > 0 else 0 for s in S]
        
        # 有效秩 (能量>5%的维度数)
        eff_rank = sum(1 for e in energy_ratios if e > 0.05)
        
        pca_results[f"L{l}"] = {
            "singular_values": [float(s) for s in S],
            "energy_ratios": [float(e) for e in energy_ratios],
            "effective_rank": eff_rank,
            "condition_number": float(S[0] / max(S[-1], 1e-10)),
        }
        
        print(f"  L{l}: S={[f'{s:.2f}' for s in S]}, "
              f"energy={[f'{e:.4f}' for e in energy_ratios]}, "
              f"eff_rank={eff_rank}, cond={S[0]/max(S[-1],1e-10):.2f}")

    # ===== 核心分析2: 旋转矩阵R的估计 =====
    print(f"\n  === Analysis 2: Rotation Matrix Estimation ===")
    rotation_results = {}
    
    # 参考层(L0)的三个Δh方向 (归一化)
    l0 = scan_layers[0]
    dh_l0 = {}
    for n in dim_names:
        dh = delta_h_all[n][l0]
        norm = np.linalg.norm(dh)
        dh_l0[n] = dh / norm if norm > 1e-8 else dh

    # 构建L0的3×d_model矩阵
    D_l0 = np.stack([dh_l0[n] for n in dim_names])  # [3, d_model]

    for l in scan_layers[1:]:  # 跳过L0
        dh_l = {}
        for n in dim_names:
            dh = delta_h_all[n][l]
            norm = np.linalg.norm(dh)
            dh_l[n] = dh / norm if norm > 1e-8 else dh

        D_l = np.stack([dh_l[n] for n in dim_names])  # [3, d_model]

        # 旋转矩阵 R: D_l ≈ R × D_l0
        # 最小二乘: R = D_l × D_l0^T × (D_l0 × D_l0^T)^{-1}
        # 但因为 D_l0 是 [3, d_model], 先投影到3D子空间
        # 更好的方法: R = D_l × D_l0^+ (伪逆)
        try:
            D_l0_pinv = np.linalg.pinv(D_l0)  # [d_model, 3]
            R = D_l @ D_l0_pinv  # [3, 3]
        except:
            R = np.eye(3)

        # 分析R的性质
        # SVD of R
        U_R, S_R, Vt_R = np.linalg.svd(R)
        
        # R的有效秩
        total_s = np.sum(S_R)
        rank_ratio = [s / total_s for s in S_R] if total_s > 0 else [0, 0, 0]
        eff_rank_R = sum(1 for r in rank_ratio if r > 0.05)
        
        # R是否接近正交? R^T R ≈ I?
        RtR = R.T @ R
        ortho_error = np.linalg.norm(RtR - np.eye(3)) / np.linalg.norm(np.eye(3))
        
        # R的行列式 (正交矩阵 det=±1)
        det_R = np.linalg.det(R)
        
        # R偏离正交矩阵的程度
        # 最近正交矩阵: U_R @ diag(1,1,det_sign) @ Vt_R
        det_sign = np.sign(det_R) if abs(det_R) > 1e-10 else 1.0
        R_ortho = U_R @ np.diag([1, 1, det_sign]) @ Vt_R
        rotation_error = np.linalg.norm(R - R_ortho) / max(np.linalg.norm(R), 1e-10)
        
        rotation_results[f"L{l}"] = {
            "R_singular_values": [float(s) for s in S_R],
            "R_rank_ratios": [float(r) for r in rank_ratio],
            "R_effective_rank": eff_rank_R,
            "R_ortho_error": float(ortho_error),
            "R_det": float(det_R),
            "R_rotation_error": float(rotation_error),
            "R_matrix": R.tolist(),
        }
        
        # 判定: 混合 vs 旋转
        if eff_rank_R < 3:
            mechanism = "MIXING (秩降低)"
        elif ortho_error < 0.3 and abs(abs(det_R) - 1.0) < 0.5:
            mechanism = "ROTATION (近正交旋转)"
        elif rotation_error < 0.3:
            mechanism = "ROTATION+SCALING (旋转+缩放)"
        else:
            mechanism = "MIXING+ROTATION (混合旋转)"
        
        rotation_results[f"L{l}"]["mechanism"] = mechanism
        
        print(f"  L{l}: S_R={[f'{s:.3f}' for s in S_R]}, "
              f"eff_rank={eff_rank_R}, det={det_R:.3f}, "
              f"ortho_err={ortho_error:.3f}, rot_err={rotation_error:.3f} → {mechanism}")

    # ===== 核心分析3: 子空间夹角分析 =====
    print(f"\n  === Analysis 3: Subspace Angles ===")
    subspace_results = {}
    
    for l in scan_layers:
        # 三个维度方向的归一化版本
        dirs = {}
        for n in dim_names:
            dh = delta_h_all[n][l]
            norm = np.linalg.norm(dh)
            dirs[n] = dh / norm if norm > 1e-8 else dh
        
        # 维度间cos (与P384一致, 用于对照)
        cos_pairs = {}
        for i, n1 in enumerate(dim_names):
            for j, n2 in enumerate(dim_names):
                if i < j:
                    cos_pairs[f"cos({n1},{n2})"] = float(np.dot(dirs[n1], dirs[n2]))
        
        # 每个维度方向在W_lm方向的投影
        wlm_projections = {}
        for n in dim_names:
            cos_wlm = float(np.dot(dirs[n], dim_directions[n]["direction"]))
            wlm_projections[f"{n}_cos_wlm"] = cos_wlm
        
        # Δh范数 (绝对信号强度)
        delta_norms = {}
        for n in dim_names:
            delta_norms[f"{n}_norm"] = float(np.linalg.norm(delta_h_all[n][l]))
        
        subspace_results[f"L{l}"] = {
            "cos_pairs": cos_pairs,
            "wlm_projections": wlm_projections,
            "delta_norms": delta_norms,
        }
        
        print(f"  L{l}: {cos_pairs}, norms=[{', '.join(f'{delta_norms[k]:.2f}' for k in sorted(delta_norms.keys()))}]")

    results = {
        "pca_results": pca_results,
        "rotation_results": rotation_results,
        "subspace_results": subspace_results,
        "scan_layers": scan_layers,
    }
    return results


# ========== P388: 交互项精确公式 ==========

def run_p388(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P388: Interaction term formula - {model_name}")
    print(f"{'='*60}")

    embed = model.get_input_embeddings()
    dim_names = ["style", "logic", "grammar"]
    dim_pairs = [STYLE_PAIRS[0], LOGIC_PAIRS[0], GRAMMAR_PAIRS[0]]

    # 计算维度方向
    dim_directions = {}
    for name, (pos, neg) in zip(dim_names, dim_pairs):
        direction, norm = get_dimension_direction(model, tokenizer, pos, neg)
        dim_directions[name] = {"direction": direction, "norm": norm, "pos": pos, "neg": neg}

    toks = tokenizer(PROMPT, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    seq_len = input_ids.shape[1]
    inputs_embeds_base = embed(input_ids).detach().clone().to(model.dtype)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    # 基线logits
    with torch.no_grad():
        logits_base = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids).logits[0, -1, :]

    # ===== 精细β扫描 =====
    print(f"\n  === Fine beta sweep ===")
    betas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0]
    
    sweep_results = {}
    
    for beta in betas:
        beta_data = {"beta": beta}
        
        # 单维度注入效果
        single_effects = {}
        for name in dim_names:
            direction = dim_directions[name]["direction"]
            w_tensor = torch.tensor(direction * beta, dtype=torch.float32, device=device)
            inputs_embeds_int = inputs_embeds_base.clone()
            inputs_embeds_int[0, -1, :] += w_tensor.to(model.dtype)
            
            with torch.no_grad():
                logits_int = model(inputs_embeds=inputs_embeds_int, position_ids=position_ids).logits[0, -1, :]
            
            pos_id = tokenizer.encode(dim_directions[name]["pos"], add_special_tokens=False)[0]
            neg_id = tokenizer.encode(dim_directions[name]["neg"], add_special_tokens=False)[0]
            delta_pos = float(logits_int[pos_id].cpu() - logits_base[pos_id].cpu())
            delta_neg = float(logits_int[neg_id].cpu() - logits_base[neg_id].cpu())
            single_effects[name] = delta_pos - delta_neg
        
        # 三维度联合注入
        combined_dir = sum(dim_directions[n]["direction"] * beta for n in dim_names)
        w_tensor = torch.tensor(combined_dir, dtype=torch.float32, device=device)
        inputs_embeds_int = inputs_embeds_base.clone()
        inputs_embeds_int[0, -1, :] += w_tensor.to(model.dtype)
        
        with torch.no_grad():
            logits_int = model(inputs_embeds=inputs_embeds_int, position_ids=position_ids).logits[0, -1, :]
        
        combined_effects = {}
        interactions = {}
        for name in dim_names:
            pos_id = tokenizer.encode(dim_directions[name]["pos"], add_special_tokens=False)[0]
            neg_id = tokenizer.encode(dim_directions[name]["neg"], add_special_tokens=False)[0]
            delta_pos = float(logits_int[pos_id].cpu() - logits_base[pos_id].cpu())
            delta_neg = float(logits_int[neg_id].cpu() - logits_base[neg_id].cpu())
            combined = delta_pos - delta_neg
            combined_effects[name] = combined
            # 交互项 = 联合效果 - 单独效果
            # 注意: 这里的"线性叠加"就是单独效果(因为维度正交, 线性叠加=单独效果之和)
            interactions[name] = combined - single_effects[name]
        
        beta_data["single"] = single_effects
        beta_data["combined"] = combined_effects
        beta_data["interaction"] = interactions
        
        sweep_results[f"beta_{beta}"] = beta_data
        
        print(f"  beta={beta:5.1f}: " + ", ".join(
            f"{n}: S={single_effects[n]:6.2f}, C={combined_effects[n]:6.2f}, I={interactions[n]:6.2f}"
            for n in dim_names
        ))

    # ===== 候选公式拟合 =====
    print(f"\n  === Formula Fitting ===")
    
    # 提取β和交互项数据
    beta_arr = np.array(betas)
    interaction_data = {}
    for n in dim_names:
        inter_values = [sweep_results[f"beta_{b}"]["interaction"][n] for b in betas]
        interaction_data[n] = np.array(inter_values)
    
    fitting_results = {}
    
    for n in dim_names:
        y = interaction_data[n]
        
        # 候选公式
        candidates = {}
        
        # 1. 线性: y = a*β
        try:
            a_lin = np.dot(beta_arr, y) / np.dot(beta_arr, beta_arr)
            y_pred_lin = a_lin * beta_arr
            r2_lin = 1 - np.sum((y - y_pred_lin)**2) / max(np.sum((y - np.mean(y))**2), 1e-10)
            candidates["linear"] = {"a": float(a_lin), "R2": float(r2_lin)}
        except:
            candidates["linear"] = {"a": 0, "R2": -999}
        
        # 2. 二次: y = a*β^2
        try:
            beta_sq = beta_arr ** 2
            a_quad = np.dot(beta_sq, y) / np.dot(beta_sq, beta_sq)
            y_pred_quad = a_quad * beta_sq
            r2_quad = 1 - np.sum((y - y_pred_quad)**2) / max(np.sum((y - np.mean(y))**2), 1e-10)
            candidates["quadratic"] = {"a": float(a_quad), "R2": float(r2_quad)}
        except:
            candidates["quadratic"] = {"a": 0, "R2": -999}
        
        # 3. S型(Michaelis-Menten): y = a*β/(1+b*β)
        try:
            from scipy.optimize import curve_fit
            def sigmoid_func(x, a, b):
                return a * x / (1 + b * x)
            popt, _ = curve_fit(sigmoid_func, beta_arr, y, p0=[1.0, 0.1], maxfev=5000)
            y_pred_sig = sigmoid_func(beta_arr, *popt)
            r2_sig = 1 - np.sum((y - y_pred_sig)**2) / max(np.sum((y - np.mean(y))**2), 1e-10)
            candidates["sigmoid"] = {"a": float(popt[0]), "b": float(popt[1]), "R2": float(r2_sig)}
        except Exception as e:
            candidates["sigmoid"] = {"a": 0, "b": 0, "R2": -999, "error": str(e)}
        
        # 4. 指数饱和: y = a*(1-exp(-b*β))
        try:
            def exp_sat_func(x, a, b):
                return a * (1 - np.exp(-b * x))
            popt, _ = curve_fit(exp_sat_func, beta_arr, y, p0=[5.0, 0.1], maxfev=5000)
            y_pred_exp = exp_sat_func(beta_arr, *popt)
            r2_exp = 1 - np.sum((y - y_pred_exp)**2) / max(np.sum((y - np.mean(y))**2), 1e-10)
            candidates["exp_saturation"] = {"a": float(popt[0]), "b": float(popt[1]), "R2": float(r2_exp)}
        except Exception as e:
            candidates["exp_saturation"] = {"a": 0, "b": 0, "R2": -999, "error": str(e)}
        
        # 5. 高斯(先增后减): y = a*β*exp(-b*β^2)
        try:
            def gauss_func(x, a, b):
                return a * x * np.exp(-b * x**2)
            # 估计初始值
            max_idx = np.argmax(np.abs(y))
            if max_idx > 0 and abs(y[max_idx]) > 0.01:
                a0 = y[max_idx] / (beta_arr[max_idx] * np.exp(-0.01 * beta_arr[max_idx]**2))
                b0 = 0.01
            else:
                a0 = 1.0
                b0 = 0.01
            popt, _ = curve_fit(gauss_func, beta_arr, y, p0=[a0, b0], maxfev=5000)
            y_pred_gauss = gauss_func(beta_arr, *popt)
            r2_gauss = 1 - np.sum((y - y_pred_gauss)**2) / max(np.sum((y - np.mean(y))**2), 1e-10)
            candidates["gaussian"] = {"a": float(popt[0]), "b": float(popt[1]), "R2": float(r2_gauss)}
        except Exception as e:
            candidates["gaussian"] = {"a": 0, "b": 0, "R2": -999, "error": str(e)}
        
        # 6. 多项式: y = a*β + b*β^2 + c*β^3
        try:
            coeffs = np.polyfit(beta_arr, y, 3)
            y_pred_poly = np.polyval(coeffs, beta_arr)
            r2_poly = 1 - np.sum((y - y_pred_poly)**2) / max(np.sum((y - np.mean(y))**2), 1e-10)
            candidates["polynomial3"] = {
                "a": float(coeffs[2]), "b": float(coeffs[1]), "c": float(coeffs[0]),
                "R2": float(r2_poly)
            }
        except:
            candidates["polynomial3"] = {"a": 0, "b": 0, "c": 0, "R2": -999}
        
        # 选最佳
        best_name = max(candidates.keys(), key=lambda k: candidates[k].get("R2", -999))
        
        fitting_results[n] = {
            "candidates": candidates,
            "best_formula": best_name,
            "best_R2": candidates[best_name].get("R2", -999),
        }
        
        print(f"\n  {n} fitting:")
        for fname, fdata in candidates.items():
            r2 = fdata.get("R2", -999)
            params = {k: v for k, v in fdata.items() if k != "R2" and k != "error"}
            print(f"    {fname}: R2={r2:.4f}, params={params}")
        print(f"    BEST: {best_name} (R2={candidates[best_name].get('R2', -999):.4f})")

    # ===== 交叉验证 =====
    print(f"\n  === Cross-validation ===")
    cv_results = {}
    
    for n in dim_names:
        y = interaction_data[n]
        # 50/50 分割
        n_half = len(beta_arr) // 2
        beta_train = beta_arr[:n_half]
        beta_test = beta_arr[n_half:]
        y_train = y[:n_half]
        y_test = y[n_half:]
        
        # 用最佳公式在训练集上拟合, 在测试集上验证
        best_name = fitting_results[n]["best_formula"]
        
        try:
            from scipy.optimize import curve_fit
            
            if best_name == "sigmoid":
                func = lambda x, a, b: a * x / (1 + b * x)
                popt, _ = curve_fit(func, beta_train, y_train, p0=[1.0, 0.1], maxfev=5000)
                y_pred_test = func(beta_test, *popt)
            elif best_name == "exp_saturation":
                func = lambda x, a, b: a * (1 - np.exp(-b * x))
                popt, _ = curve_fit(func, beta_train, y_train, p0=[5.0, 0.1], maxfev=5000)
                y_pred_test = func(beta_test, *popt)
            elif best_name == "gaussian":
                func = lambda x, a, b: a * x * np.exp(-b * x**2)
                popt, _ = curve_fit(func, beta_train, y_train, p0=[1.0, 0.01], maxfev=5000)
                y_pred_test = func(beta_test, *popt)
            elif best_name == "polynomial3":
                coeffs = np.polyfit(beta_train, y_train, 3)
                y_pred_test = np.polyval(coeffs, beta_test)
            elif best_name == "quadratic":
                beta_sq = beta_train ** 2
                a = np.dot(beta_sq, y_train) / np.dot(beta_sq, beta_sq)
                y_pred_test = a * beta_test ** 2
            else:  # linear
                a = np.dot(beta_train, y_train) / np.dot(beta_train, beta_train)
                y_pred_test = a * beta_test
            
            cv_r2 = 1 - np.sum((y_test - y_pred_test)**2) / max(np.sum((y_test - np.mean(y_test))**2), 1e-10)
            cv_results[n] = {"best_formula": best_name, "CV_R2": float(cv_r2)}
            print(f"  {n}: best={best_name}, CV_R2={cv_r2:.4f}")
        except Exception as e:
            cv_results[n] = {"best_formula": best_name, "CV_R2": -999, "error": str(e)}
            print(f"  {n}: CV failed: {e}")

    results = {
        "sweep_results": sweep_results,
        "fitting_results": fitting_results,
        "cv_results": cv_results,
        "betas": betas,
    }
    return results


# ========== P389: W_up训练动力学推断 ==========

def run_p389(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P389: W_up training dynamics inference - {model_name}")
    print(f"{'='*60}")

    embed = model.get_input_embeddings()
    n_layers_total = len(get_layers(model))
    hidden_dim = embed.weight.shape[1]
    layers = get_layers(model)

    # ===== 1. 每层MLP权重谱范数 =====
    print(f"\n  === Per-layer MLP spectral norms ===")
    layer_stats = []
    
    for i, layer in enumerate(layers):
        mlp = get_mlp(model, layer)
        stats = {"layer": i}
        
        # W_up
        if hasattr(mlp, "up_proj"):
            W_up = mlp.up_proj.weight.detach().cpu().float()
        elif hasattr(mlp, "gate_up_proj"):
            W_full = mlp.gate_up_proj.weight.detach().cpu().float()
            inter_size = W_full.shape[0] // 2
            W_up = W_full[inter_size:, :]
        else:
            W_up = None
        
        if W_up is not None:
            try:
                _, S_up, _ = torch.linalg.svd(W_up, full_matrices=False)
                stats["W_up_sigma_max"] = float(S_up[0].item())
                stats["W_up_sigma_min"] = float(S_up[-1].item())
            except:
                stats["W_up_sigma_max"] = float(W_up.norm().item())
                stats["W_up_sigma_min"] = 0.0
            stats["W_up_frobenius"] = float(W_up.norm().item())
            stats["W_up_shape"] = list(W_up.shape)
        
        # W_gate
        if hasattr(mlp, "gate_proj"):
            W_gate = mlp.gate_proj.weight.detach().cpu().float()
        elif hasattr(mlp, "gate_up_proj"):
            W_full = mlp.gate_up_proj.weight.detach().cpu().float()
            inter_size = W_full.shape[0] // 2
            W_gate = W_full[:inter_size, :]
        else:
            W_gate = None
        
        if W_gate is not None:
            try:
                _, S_gate, _ = torch.linalg.svd(W_gate, full_matrices=False)
                stats["W_gate_sigma_max"] = float(S_gate[0].item())
            except:
                stats["W_gate_sigma_max"] = float(W_gate.norm().item())
            stats["W_gate_frobenius"] = float(W_gate.norm().item())
        
        # W_down
        if hasattr(mlp, "down_proj"):
            W_down = mlp.down_proj.weight.detach().cpu().float()
        elif hasattr(mlp, "dense"):
            W_down = mlp.dense.weight.detach().cpu().float()
        else:
            W_down = None
        
        if W_down is not None:
            try:
                _, S_down, _ = torch.linalg.svd(W_down, full_matrices=False)
                stats["W_down_sigma_max"] = float(S_down[0].item())
            except:
                stats["W_down_sigma_max"] = float(W_down.norm().item())
            stats["W_down_frobenius"] = float(W_down.norm().item())
        
        layer_stats.append(stats)
        
        if i < 10 or i == n_layers_total - 1:
            print(f"  L{i}: W_up_σmax={stats.get('W_up_sigma_max', 0):.3f}, "
                  f"W_gate_σmax={stats.get('W_gate_sigma_max', 0):.3f}, "
                  f"W_down_σmax={stats.get('W_down_sigma_max', 0):.3f}")

    # ===== 2. 层间相关性 =====
    print(f"\n  === Inter-layer correlations ===")
    
    w_up_sigmas = [s.get("W_up_sigma_max", 0) for s in layer_stats]
    w_gate_sigmas = [s.get("W_gate_sigma_max", 0) for s in layer_stats]
    w_down_sigmas = [s.get("W_down_sigma_max", 0) for s in layer_stats]
    
    # 自相关: σ_up[l] vs σ_up[l+1]
    if len(w_up_sigmas) > 1:
        auto_corr_up = np.corrcoef(w_up_sigmas[:-1], w_up_sigmas[1:])[0, 1]
    else:
        auto_corr_up = 0
    
    # 交叉相关: σ_up vs σ_gate
    if len(w_up_sigmas) > 0 and len(w_gate_sigmas) > 0 and any(g > 0 for g in w_gate_sigmas):
        cross_corr_up_gate = np.corrcoef(w_up_sigmas, w_gate_sigmas)[0, 1]
    else:
        cross_corr_up_gate = 0
    
    # 交叉相关: σ_up vs σ_down
    if len(w_up_sigmas) > 0 and len(w_down_sigmas) > 0 and any(d > 0 for d in w_down_sigmas):
        cross_corr_up_down = np.corrcoef(w_up_sigmas, w_down_sigmas)[0, 1]
    else:
        cross_corr_up_down = 0
    
    print(f"  Auto-corr(σ_up[l], σ_up[l+1]) = {auto_corr_up:.4f}")
    print(f"  Cross-corr(σ_up, σ_gate) = {cross_corr_up_gate:.4f}")
    print(f"  Cross-corr(σ_up, σ_down) = {cross_corr_up_down:.4f}")

    # ===== 3. W_up分布模式分析 =====
    print(f"\n  === W_up distribution pattern ===")
    
    w_up_arr = np.array(w_up_sigmas)
    
    # 趋势分析
    layer_indices = np.arange(len(w_up_arr))
    if len(w_up_arr) > 2:
        slope, intercept = np.polyfit(layer_indices, w_up_arr, 1)
        # 相对变化
        w_up_min = np.min(w_up_arr)
        w_up_max = np.max(w_up_arr)
        w_up_range = w_up_max - w_up_min
        w_up_mean = np.mean(w_up_arr)
        w_up_cv = np.std(w_up_arr) / max(w_up_mean, 1e-10)  # 变异系数
        
        # 分段统计
        n_third = len(w_up_arr) // 3
        early_mean = np.mean(w_up_arr[:n_third]) if n_third > 0 else w_up_arr[0]
        mid_mean = np.mean(w_up_arr[n_third:2*n_third]) if n_third > 0 else w_up_arr[0]
        late_mean = np.mean(w_up_arr[2*n_third:]) if n_third > 0 else w_up_arr[-1]
        
        # U型/倒U型检测
        quad_coeffs = np.polyfit(layer_indices, w_up_arr, 2)
        is_U_shape = quad_coeffs[0] > 0  # 正二次系数 = U型
        is_inv_U_shape = quad_coeffs[0] < 0  # 负二次系数 = 倒U型
        
        pattern = "unknown"
        if abs(quad_coeffs[0]) < 0.001:
            pattern = "flat (均匀)"
        elif is_U_shape:
            pattern = "U-shape (首尾大, 中间小)"
        elif is_inv_U_shape:
            pattern = "inverted-U (中间大, 首尾小)"
        
        if abs(slope) > 0.01 and abs(quad_coeffs[0]) < abs(slope) * 0.1:
            pattern = "increasing" if slope > 0 else "decreasing"
        
        print(f"  W_up trend: slope={slope:.4f}, quad_coeff={quad_coeffs[0]:.6f}")
        print(f"  W_up range: [{w_up_min:.3f}, {w_up_max:.3f}], mean={w_up_mean:.3f}, CV={w_up_cv:.4f}")
        print(f"  W_up by phase: early={early_mean:.3f}, mid={mid_mean:.3f}, late={late_mean:.3f}")
        print(f"  Pattern: {pattern}")
    else:
        slope = 0
        pattern = "insufficient_data"
        w_up_cv = 0
        quad_coeffs = [0, 0, 0]

    # ===== 4. W_up与注意力维度的关系 =====
    print(f"\n  === W_up vs Attention dimension ===")
    
    attn_stats = []
    for i, layer in enumerate(layers):
        if hasattr(layer, "self_attn"):
            attn = layer.self_attn
        elif hasattr(layer, "attention"):
            attn = layer.attention
        else:
            attn_stats.append({"layer": i})
            continue
        
        stats = {"layer": i}
        
        # Q/K/V投影的谱范数
        if hasattr(attn, "q_proj"):
            W_q = attn.q_proj.weight.detach().cpu().float()
            stats["W_q_sigma_max"] = float(torch.linalg.svdvals(W_q)[0].item())
        if hasattr(attn, "k_proj"):
            W_k = attn.k_proj.weight.detach().cpu().float()
            stats["W_k_sigma_max"] = float(torch.linalg.svdvals(W_k)[0].item())
        if hasattr(attn, "v_proj"):
            W_v = attn.v_proj.weight.detach().cpu().float()
            stats["W_v_sigma_max"] = float(torch.linalg.svdvals(W_v)[0].item())
        
        attn_stats.append(stats)
    
    # W_up与W_q的相关性
    w_q_sigmas = [s.get("W_q_sigma_max", 0) for s in attn_stats]
    w_v_sigmas = [s.get("W_v_sigma_max", 0) for s in attn_stats]
    
    if len(w_up_sigmas) > 2 and len(w_q_sigmas) > 2 and any(q > 0 for q in w_q_sigmas):
        corr_up_q = np.corrcoef(w_up_sigmas, w_q_sigmas)[0, 1]
        corr_up_v = np.corrcoef(w_up_sigmas, w_v_sigmas)[0, 1] if any(v > 0 for v in w_v_sigmas) else 0
    else:
        corr_up_q = 0
        corr_up_v = 0
    
    print(f"  Corr(σ_up, σ_q) = {corr_up_q:.4f}")
    print(f"  Corr(σ_up, σ_v) = {corr_up_v:.4f}")

    # ===== 5. W_up与方向保持度的关系(推断) =====
    print(f"\n  === W_up vs Direction Preservation (inferred) ===")
    
    # 根据已有理论: cos(t) = cos(0) * exp(-λ * Σσ_up)
    # 半衰期 = ln2 / (λ * σ_up_avg)
    # λ ≈ 0.022 / W_lm_norm
    
    lm_head = model.lm_head
    W_lm = lm_head.weight.detach().cpu().float()
    W_lm_mean_norm = float(W_lm.norm(dim=1).mean())
    lam = 0.022 / W_lm_mean_norm
    
    sigma_up_avg = np.mean(w_up_sigmas) if len(w_up_sigmas) > 0 else 0
    half_life = np.log(2) / (lam * sigma_up_avg) if lam * sigma_up_avg > 0 else float('inf')
    
    print(f"  λ ≈ {lam:.6f}")
    print(f"  σ_up_avg = {sigma_up_avg:.4f}")
    print(f"  Predicted half-life = {half_life:.1f} layers")
    print(f"  Model total layers = {n_layers_total}")
    print(f"  Coverage = half_life / n_layers = {half_life/n_layers_total:.2f}" if n_layers_total > 0 else "  N/A")

    results = {
        "layer_stats": layer_stats,
        "attn_stats": attn_stats,
        "auto_corr_up": float(auto_corr_up),
        "cross_corr_up_gate": float(cross_corr_up_gate),
        "cross_corr_up_down": float(cross_corr_up_down),
        "w_up_trend_slope": float(slope),
        "w_up_quad_coeff": float(quad_coeffs[0]),
        "w_up_pattern": pattern,
        "w_up_cv": float(w_up_cv),
        "corr_up_q": float(corr_up_q),
        "corr_up_v": float(corr_up_v),
        "inferred_lambda": float(lam),
        "inferred_half_life": float(half_life),
    }
    return results


# ========== Main ==========

def main():
    parser = argparse.ArgumentParser(description="Phase LXXVI: Rotation/Mixing + Interaction Formula + W_up Dynamics")
    parser.add_argument("--model", type=str, default="qwen3", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=str, default="all", choices=["p387", "p388", "p389", "all"])
    args = parser.parse_args()

    models_to_run = ["qwen3", "glm4", "deepseek7b"] if args.model == "qwen3" else [args.model]

    for model_name in models_to_run:
        print(f"\n{'#'*70}")
        print(f"# Testing model: {model_name}")
        print(f"{'#'*70}")

        model, tokenizer, device = load_model(model_name)
        timestamp = time.strftime("%Y%m%d_%H%M")

        all_results = {"model": model_name, "timestamp": timestamp}

        try:
            if args.exp in ["p387", "all"]:
                r387 = run_p387(model, tokenizer, device, model_name)
                all_results["p387"] = r387

            if args.exp in ["p388", "all"]:
                r388 = run_p388(model, tokenizer, device, model_name)
                all_results["p388"] = r388

            if args.exp in ["p389", "all"]:
                r389 = run_p389(model, tokenizer, device, model_name)
                all_results["p389"] = r389

        except Exception as e:
            print(f"  ERROR in {model_name}: {e}")
            traceback.print_exc()
            all_results["error"] = str(e)

        # Save
        out_file = OUT_DIR / f"phase_lxxvi_p387_389_{model_name}_{timestamp}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nResults saved: {out_file}")

        # Release GPU
        del model
        torch.cuda.empty_cache()
        print(f"GPU memory released, waiting 5s...")
        time.sleep(5)


if __name__ == "__main__":
    main()
