#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage628-629-630: GLM4消歧隐藏通道 + Hidden-Logit统一方程 + 逐层消歧贡献

Stage628: GLM4消歧的隐藏通道分析（P0）
  原理：GLM4 hidden消歧度最高(70%)但logit margin最小(0.10)，说明消歧信息编码在
  unembed无法高效读出的方向上。需要分析：
  - unembed矩阵对消歧token的行向量与消歧方向的余弦相似度
  - 对比GLM4与其他模型的unembed行向量分布
  - 分析消歧方向在unembed主子空间vs次子空间的投影
  - "隐藏通道"：消歧信息存在但unembed读出效率低的方向

Stage629: Hidden→Logit消歧映射的统一方程（P0）
  原理：hidden cos_dist和logit margin不一致(|corr|<0.3)，需要理解为什么。
  - 消歧方向 d = h_A - h_B（hidden空间）
  - Logit margin = d · (W_unembed[tok_a] - W_unembed[tok_b])
  - 分析 d 与 unembed行差向量的关系
  - 分解消歧信息：多少被unembed读出(logit margin)，多少被"浪费"(投影到低敏感方向)
  - 建立统一方程：hidden_disamb = f(logit_disamb, wasted_component)

Stage630: 逐层消歧贡献因果实验（P1）
  原理：逐层零化MLP+Attn，测量每层对末层消歧的贡献。
  - baseline: 正常前向传播，测量末层消歧度
  - 对每层l: 零化MLP(l)，重新前向传播，测量消歧度变化Δ_l
  - 对每层l: 零化Attn(l)，重新前向传播，测量消歧度变化
  - 找出"关键层"（对消歧贡献最大的层）
  - 比较四模型的关键层分布

用法: python stage628_629_630_glm4_hidden_channel.py [qwen3|deepseek7b|glm4|gemma4]
"""

from __future__ import annotations
import sys, json, time, gc, torch, os, copy
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "codex"))

from multimodel_language_shared import (
    load_model_bundle, free_model, discover_layers, encode_to_device
)

OUTPUT_DIR = PROJECT_ROOT / "tests" / "glm5_temp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")


def safe_get_device(model):
    for attr in [None, 'model', 'model.model']:
        try:
            obj = model
            if attr:
                for part in attr.split('.'):
                    obj = getattr(obj, part)
            return next(obj.parameters()).device
        except (StopIteration, AttributeError):
            continue
    return torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")


def move_to_device(batch, model):
    device = safe_get_device(model)
    if hasattr(batch, 'to'):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in batch.items()}
    return batch


def cos_sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def extract_all_layer_hidden(model, tokenizer, sentence, layers):
    """提取每层最后一个token的hidden state"""
    device = safe_get_device(model)
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
    inputs = move_to_device(inputs, model)

    hidden_states = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            if h.dim() >= 2:
                hidden_states[layer_idx] = h[0, -1, :].float().detach().cpu()
        return hook_fn

    for li, layer_module in enumerate(layers):
        hooks.append(layer_module.register_forward_hook(make_hook(li)))

    try:
        with torch.no_grad():
            model(**inputs)
    except Exception as e:
        print(f"    Forward failed: {e}")
    finally:
        for h in hooks:
            h.remove()

    if len(hidden_states) == 0:
        return None
    return hidden_states


def get_unembed_matrix(model):
    """获取unembedding矩阵"""
    for attr_path in ['lm_head', 'embed_out', 'output']:
        try:
            w = getattr(model, attr_path, None)
            if w is not None and hasattr(w, 'weight'):
                return w.weight.float().detach().cpu()
        except:
            pass
    try:
        inner = getattr(model, 'model', model)
        for attr_path in ['lm_head', 'embed_out', 'output']:
            try:
                w = getattr(inner, attr_path, None)
                if w is not None and hasattr(w, 'weight'):
                    return w.weight.float().detach().cpu()
            except:
                pass
    except:
        pass
    return None


def get_layer_modules(layers):
    """获取MLP和Attn模块列表"""
    mlp_modules = []
    attn_modules = []
    
    for li, layer in enumerate(layers):
        mlp_found = None
        attn_found = None
        for name, child in layer.named_children():
            nl = name.lower()
            if 'mlp' in nl or 'feed_forward' in nl or 'ffn' in nl:
                mlp_found = child
            if 'attn' in nl or 'attention' in nl or 'self_attn' in nl:
                attn_found = child
        mlp_modules.append((li, mlp_found))
        attn_modules.append((li, attn_found))
    
    return mlp_modules, attn_modules


DISAMB_PAIRS = [
    ("The river bank was muddy.", "The bank approved the loan.", "bank",
     "bank", "financial"),
    ("She ate a red apple.", "Apple released the iPhone.", "apple",
     "apple", "Apple"),
    ("The factory plant employs workers.", "She watered the plant.", "plant",
     "plant", "factory"),
    ("He went to the bank to deposit money.", "The river bank was steep.", "bank2",
     "bank", "river"),
    ("The spring water was cold.", "The metal spring bounced back.", "spring",
     "spring", "season"),
]


# ============ Stage628: GLM4消歧的隐藏通道分析 ============

def run_stage628(model, tokenizer, model_key):
    """分析消歧信息在unembed主/次子空间的分布"""
    print("\n" + "="*70)
    print("Stage628: 消歧信息在unembed子空间的分布（隐藏通道分析）")
    print("="*70)
    
    W_unembed = get_unembed_matrix(model)
    if W_unembed is None:
        print("  [SKIP] 无法获取unembed矩阵")
        return {}
    
    layers = discover_layers(model)
    n_layers = len(layers)
    hidden_dim = W_unembed.shape[1]
    
    # SVD of unembed matrix
    U, S, Vh = torch.linalg.svd(W_unembed.float(), full_matrices=False)
    # Vh rows = right singular vectors (hidden space directions)
    # U columns = left singular vectors (vocab space directions)
    
    # Split into principal (top-k) and residual (rest) subspaces
    k_principal = min(hidden_dim // 2, 500)  # Top half as principal
    V_principal = Vh[:k_principal, :]  # [k, hidden_dim] - principal directions
    V_residual = Vh[k_principal:, :]   # [rest, hidden_dim] - residual directions
    
    results = {}
    
    for ctx_a, ctx_b, word, tok_a_name, tok_b_name in DISAMB_PAIRS:
        tok_a = tokenizer.encode(tok_a_name, add_special_tokens=False)[-1]
        tok_b = tokenizer.encode(tok_b_name, add_special_tokens=False)[-1]
        
        h_a_layers = extract_all_layer_hidden(model, tokenizer, ctx_a, layers)
        h_b_layers = extract_all_layer_hidden(model, tokenizer, ctx_b, layers)
        
        if not h_a_layers or not h_b_layers:
            continue
        
        pair_data = []
        
        for li in sorted(h_a_layers.keys()):
            h_a = h_a_layers[li].float()
            h_b = h_b_layers[li].float()
            
            if h_a.shape != h_b.shape:
                continue
            
            # Disambiguation direction
            d = h_a - h_b
            d_norm = torch.norm(d).item()
            if d_norm < 1e-8:
                continue
            d_unit = d / d_norm
            
            # Project d onto unembed singular vectors
            # d · Vh[i] gives component along i-th singular direction
            sv_components = (d_unit @ Vh.T)  # [hidden_dim]
            
            # Energy in principal vs residual subspace
            principal_energy = (sv_components[:k_principal] ** 2).sum().item()
            residual_energy = (sv_components[k_principal:] ** 2).sum().item()
            total_energy = principal_energy + residual_energy
            
            principal_ratio = principal_energy / total_energy if total_energy > 0 else 0
            
            # Logit margin
            logits_a = (h_a @ W_unembed.T)
            logits_b = (h_b @ W_unembed.T)
            margin_a = logits_a[tok_a].item() - logits_a[tok_b].item()
            margin_b = logits_b[tok_b].item() - logits_b[tok_a].item()
            logit_margin = abs((margin_a + margin_b) / 2)
            
            # Cos_sim between d and unembed row difference
            unembed_diff = W_unembed[tok_a] - W_unembed[tok_b]
            unembed_diff_norm = torch.norm(unembed_diff)
            if unembed_diff_norm > 1e-8:
                alignment = cos_sim(d_unit, unembed_diff / unembed_diff_norm)
            else:
                alignment = 0.0
            
            # Hidden cos_dist
            hidden_cos_dist = 1 - cos_sim(h_a, h_b)
            
            # Top-10 singular direction contributions
            top10_energy = (sv_components[:10] ** 2).sum().item()
            top10_ratio = top10_energy / total_energy if total_energy > 0 else 0
            
            pair_data.append({
                "layer": li,
                "principal_ratio": round(principal_ratio, 4),
                "residual_ratio": round(1 - principal_ratio, 4),
                "logit_margin": round(logit_margin, 4),
                "alignment_to_unembed_diff": round(alignment, 4),
                "hidden_cos_dist": round(hidden_cos_dist, 4),
                "top10_sv_ratio": round(top10_ratio, 4),
            })
        
        results[word] = pair_data
    
    # Summary: average across all pairs and layers
    all_principal = []
    all_residual = []
    all_alignment = []
    all_hidden_cos = []
    all_logit_margin = []
    all_top10 = []
    
    for word, data_list in results.items():
        for d in data_list:
            all_principal.append(d["principal_ratio"])
            all_residual.append(d["residual_ratio"])
            all_alignment.append(abs(d["alignment_to_unembed_diff"]))
            all_hidden_cos.append(d["hidden_cos_dist"])
            all_logit_margin.append(d["logit_margin"])
            all_top10.append(d["top10_sv_ratio"])
    
    summary = {}
    if all_principal:
        summary["mean_principal_ratio"] = round(float(np.mean(all_principal)), 4)
        summary["mean_residual_ratio"] = round(float(np.mean(all_residual)), 4)
        summary["mean_alignment"] = round(float(np.mean(all_alignment)), 4)
        summary["mean_hidden_cos"] = round(float(np.mean(all_hidden_cos)), 4)
        summary["mean_logit_margin"] = round(float(np.mean(all_logit_margin)), 4)
        summary["mean_top10_sv_ratio"] = round(float(np.mean(all_top10)), 4)
        
        # Correlation: alignment vs logit_margin
        if len(all_alignment) > 5:
            corr_align_margin = float(np.corrcoef(all_alignment, all_logit_margin)[0, 1])
            summary["alignment_margin_corr"] = round(corr_align_margin, 4)
        
        # Correlation: residual_ratio vs hidden_cos_dist
        if len(all_residual) > 5:
            corr_res_hidden = float(np.corrcoef(all_residual, all_hidden_cos)[0, 1])
            summary["residual_hidden_corr"] = round(corr_res_hidden, 4)
    
    print(f"\n  --- Stage628 摘要 ---")
    print(f"  消歧方向在unembed主子空间的能量比: {summary.get('mean_principal_ratio', 'N/A')}")
    print(f"  消歧方向在unembed次子空间的能量比: {summary.get('mean_residual_ratio', 'N/A')}")
    print(f"  Top10奇异方向承载能量比: {summary.get('mean_top10_sv_ratio', 'N/A')}")
    print(f"  消歧方向与unembed差向量对齐度: {summary.get('mean_alignment', 'N/A')}")
    print(f"  Alignment-margin相关性: {summary.get('alignment_margin_corr', 'N/A')}")
    print(f"  次子空间-hidden相关性: {summary.get('residual_hidden_corr', 'N/A')}")
    
    # Per-word final layer summary
    for word, data_list in results.items():
        if data_list:
            last = data_list[-1]
            print(f"  [{word}] 末层: principal={last['principal_ratio']:.3f}, "
                  f"align={last['alignment_to_unembed_diff']:.3f}, "
                  f"logit_m={last['logit_margin']:.3f}, cos_d={last['hidden_cos_dist']:.3f}")
    
    summary["per_pair"] = results
    summary["k_principal"] = k_principal
    return summary


# ============ Stage629: Hidden→Logit消歧映射的统一方程 ============

def run_stage629(model, tokenizer, model_key):
    """建立hidden消歧→logit消歧的统一映射方程"""
    print("\n" + "="*70)
    print("Stage629: Hidden→Logit消歧映射的统一方程")
    print("="*70)
    
    W_unembed = get_unembed_matrix(model)
    if W_unembed is None:
        print("  [SKIP] 无法获取unembed矩阵")
        return {}
    
    layers = discover_layers(model)
    hidden_dim = W_unembed.shape[1]
    
    results = {}
    
    for ctx_a, ctx_b, word, tok_a_name, tok_b_name in DISAMB_PAIRS:
        tok_a = tokenizer.encode(tok_a_name, add_special_tokens=False)[-1]
        tok_b = tokenizer.encode(tok_b_name, add_special_tokens=False)[-1]
        
        h_a_layers = extract_all_layer_hidden(model, tokenizer, ctx_a, layers)
        h_b_layers = extract_all_layer_hidden(model, tokenizer, ctx_b, layers)
        
        if not h_a_layers or not h_b_layers:
            continue
        
        pair_data = []
        
        for li in sorted(h_a_layers.keys()):
            h_a = h_a_layers[li].float()
            h_b = h_b_layers[li].float()
            
            if h_a.shape != h_b.shape:
                continue
            
            # === Core equation decomposition ===
            # logit_margin_A = h_a · W_u[tok_a] - h_a · W_u[tok_b]
            #                  = h_a · (W_u[tok_a] - W_u[tok_b])
            #                  = h_a · delta_u
            # where delta_u = W_u[tok_a] - W_u[tok_b]
            
            d = h_a - h_b  # disambiguation direction in hidden space
            d_norm = torch.norm(d).item()
            
            # Unembed difference vector
            delta_u = W_unembed[tok_a] - W_unembed[tok_b]  # [hidden_dim]
            delta_u_norm = torch.norm(delta_u).item()
            
            # Logit contributions
            # margin = (h_a · delta_u) for context A favoring tok_a
            #        - (h_b · delta_u) for context B favoring tok_b
            # actual = d · delta_u (where d = h_a - h_b)
            logit_contribution = float((d @ delta_u).item())
            
            # Decompose: what fraction of d_norm contributes to logit?
            if delta_u_norm > 1e-8:
                # Project d onto delta_u direction
                d_proj_on_delta = float((d @ delta_u).item()) / delta_u_norm
                d_perp_norm = np.sqrt(max(0, d_norm**2 - d_proj_on_delta**2))
                
                # Efficiency: how much of the hidden difference is "useful" for logit margin
                efficiency = abs(d_proj_on_delta) / d_norm if d_norm > 0 else 0
                # Cosine between d and delta_u
                cos_d_delta = float(cos_sim(d, delta_u)) if d_norm > 1e-8 and delta_u_norm > 1e-8 else 0
            else:
                d_proj_on_delta = 0
                d_perp_norm = d_norm
                efficiency = 0
                cos_d_delta = 0
            
            # Now measure actual logit margin
            logits_a = (h_a @ W_unembed.T)
            logits_b = (h_b @ W_unembed.T)
            margin_a = logits_a[tok_a].item() - logits_a[tok_b].item()
            margin_b = logits_b[tok_b].item() - logits_b[tok_a].item()
            actual_margin = (margin_a + margin_b) / 2
            
            # Hidden cos_dist
            hidden_cos = cos_sim(h_a, h_b)
            hidden_cos_dist = 1 - hidden_cos
            
            # "Wasted" component: the part of d perpendicular to delta_u
            waste_ratio = 1 - efficiency
            
            pair_data.append({
                "layer": li,
                "d_norm": round(d_norm, 4),
                "delta_u_norm": round(delta_u_norm, 4),
                "cos_d_delta": round(cos_d_delta, 4),
                "efficiency": round(efficiency, 4),
                "waste_ratio": round(waste_ratio, 4),
                "d_proj_on_delta": round(d_proj_on_delta, 4),
                "d_perp_norm": round(d_perp_norm, 4),
                "logit_contribution": round(logit_contribution, 4),
                "actual_margin": round(actual_margin, 4),
                "hidden_cos_dist": round(hidden_cos_dist, 4),
            })
        
        results[word] = pair_data
    
    # Summary
    all_efficiency = []
    all_cos_d_delta = []
    all_hidden_cos = []
    all_actual_margin = []
    all_d_norm = []
    all_waste = []
    
    for word, data_list in results.items():
        for d in data_list:
            all_efficiency.append(d["efficiency"])
            all_cos_d_delta.append(abs(d["cos_d_delta"]))
            all_hidden_cos.append(d["hidden_cos_dist"])
            all_actual_margin.append(abs(d["actual_margin"]))
            all_d_norm.append(d["d_norm"])
            all_waste.append(d["waste_ratio"])
    
    summary = {}
    if all_efficiency:
        summary["mean_efficiency"] = round(float(np.mean(all_efficiency)), 4)
        summary["mean_waste_ratio"] = round(float(np.mean(all_waste)), 4)
        summary["mean_cos_d_delta"] = round(float(np.mean(all_cos_d_delta)), 4)
        summary["mean_hidden_cos_dist"] = round(float(np.mean(all_hidden_cos)), 4)
        summary["mean_actual_margin"] = round(float(np.mean(all_actual_margin)), 4)
        summary["mean_d_norm"] = round(float(np.mean(all_d_norm)), 4)
        
        # Key correlation: efficiency vs actual_margin
        if len(all_efficiency) > 5:
            corr_eff_margin = float(np.corrcoef(all_efficiency, all_actual_margin)[0, 1])
            summary["efficiency_margin_corr"] = round(corr_eff_margin, 4)
        
        # Hidden cos_dist vs actual_margin
        if len(all_hidden_cos) > 5:
            corr_hidden_margin = float(np.corrcoef(all_hidden_cos, all_actual_margin)[0, 1])
            summary["hidden_margin_corr"] = round(corr_hidden_margin, 4)
        
        # d_norm vs actual_margin
        if len(all_d_norm) > 5:
            corr_dnorm_margin = float(np.corrcoef(all_d_norm, all_actual_margin)[0, 1])
            summary["dnorm_margin_corr"] = round(corr_dnorm_margin, 4)
        
        # Unified equation: margin ≈ d_norm × efficiency × delta_u_norm
        # Predicted margin = cos(d, delta_u) × d_norm × delta_u_norm / delta_u_norm
        #                 = cos(d, delta_u) × d_norm
        predicted_margins = [abs(c) * dn for c, dn in zip(all_cos_d_delta, all_d_norm)]
        actual_margins = all_actual_margin
        if len(predicted_margins) > 5:
            corr_pred_actual = float(np.corrcoef(predicted_margins, actual_margins)[0, 1])
            summary["predicted_actual_corr"] = round(corr_pred_actual, 4)
            summary["prediction_R2"] = round(corr_pred_actual ** 2, 4)
    
    print(f"\n  --- Stage629 摘要 ---")
    print(f"  消歧→Logit映射效率: {summary.get('mean_efficiency', 'N/A')} "
          f"(浪费: {summary.get('mean_waste_ratio', 'N/A')})")
    print(f"  消歧方向与unembed差对齐: {summary.get('mean_cos_d_delta', 'N/A')}")
    print(f"  效率-margin相关性: {summary.get('efficiency_margin_corr', 'N/A')}")
    print(f"  Hidden-margin相关性: {summary.get('hidden_margin_corr', 'N/A')}")
    print(f"  预测方程精度(R2): {summary.get('prediction_R2', 'N/A')}")
    
    # Per-word final layer
    for word, data_list in results.items():
        if data_list:
            last = data_list[-1]
            print(f"  [{word}] 末层: eff={last['efficiency']:.3f}, cos={last['cos_d_delta']:.3f}, "
                  f"margin={last['actual_margin']:.3f}, d_norm={last['d_norm']:.2f}")
    
    summary["per_pair"] = results
    
    # The unified equation
    summary["unified_equation"] = {
        "formula": "logit_margin ≈ cos(d, delta_u) × ||d|| × ||delta_u|| / ||delta_u|| = cos(d, delta_u) × ||d||",
        "where": {
            "d": "h_A - h_B (hidden space disambiguation direction)",
            "delta_u": "W_unembed[tok_a] - W_unembed[tok_b] (unembed readout direction)",
            "efficiency": "cos(d, delta_u) = how well d aligns with unembed readout",
        }
    }
    
    return summary


# ============ Stage630: 逐层消歧贡献因果实验 ============

def run_stage630(model, tokenizer, model_key):
    """逐层零化MLP/Attn，测量每层对消歧的贡献"""
    print("\n" + "="*70)
    print("Stage630: 逐层消歧贡献因果实验")
    print("="*70)
    
    W_unembed = get_unembed_matrix(model)
    if W_unembed is None:
        print("  [SKIP] 无法获取unembed矩阵")
        return {}
    
    layers = discover_layers(model)
    n_layers = len(layers)
    mlp_modules, attn_modules = get_layer_modules(layers)
    
    device = safe_get_device(model)
    
    def measure_final_disamb(model_ref):
        """测量末层消歧度（logit margin + hidden cos_dist）"""
        logit_margins = []
        hidden_cos_dists = []
        
        for ctx_a, ctx_b, word, tok_a_name, tok_b_name in DISAMB_PAIRS:
            tok_a = tokenizer.encode(tok_a_name, add_special_tokens=False)[-1]
            tok_b = tokenizer.encode(tok_b_name, add_special_tokens=False)[-1]
            
            h_a_layers = extract_all_layer_hidden(model_ref, tokenizer, ctx_a, layers)
            h_b_layers = extract_all_layer_hidden(model_ref, tokenizer, ctx_b, layers)
            
            if not h_a_layers or not h_b_layers:
                continue
            
            # Use last layer
            last_li = max(h_a_layers.keys())
            h_a = h_a_layers[last_li].float()
            h_b = h_b_layers[last_li].float()
            
            if h_a.shape != h_b.shape:
                continue
            
            # Logit margin
            logits_a = (h_a @ W_unembed.T)
            logits_b = (h_b @ W_unembed.T)
            margin_a = logits_a[tok_a].item() - logits_a[tok_b].item()
            margin_b = logits_b[tok_b].item() - logits_b[tok_a].item()
            logit_margins.append(abs((margin_a + margin_b) / 2))
            
            # Hidden cos_dist
            hidden_cos_dists.append(1 - cos_sim(h_a, h_b))
        
        return {
            "mean_logit_margin": float(np.mean(logit_margins)) if logit_margins else 0,
            "mean_hidden_cos_dist": float(np.mean(hidden_cos_dists)) if hidden_cos_dists else 0,
        }
    
    def zero_mlp_at_layer(layer_idx, model_ref):
        """临时零化指定层的MLP输出"""
        hooks = []
        li, mlp_mod = mlp_modules[layer_idx]
        if mlp_mod is None:
            return hooks
        
        def zero_hook(module, input, output):
            if isinstance(output, tuple):
                return (torch.zeros_like(output[0]),) + output[1:]
            return torch.zeros_like(output)
        
        h = mlp_mod.register_forward_hook(zero_hook)
        hooks.append(h)
        return hooks
    
    def zero_attn_at_layer(layer_idx, model_ref):
        """临时零化指定层的Attn输出"""
        hooks = []
        li, attn_mod = attn_modules[layer_idx]
        if attn_mod is None:
            return hooks
        
        def zero_hook(module, input, output):
            if isinstance(output, tuple):
                if len(output) >= 2:
                    return (torch.zeros_like(output[0]),) + output[1:]
                return (torch.zeros_like(output[0]),)
            return torch.zeros_like(output)
        
        h = attn_mod.register_forward_hook(zero_hook)
        hooks.append(h)
        return hooks
    
    def remove_hooks(hooks):
        for h in hooks:
            h.remove()
    
    # Baseline
    print("  测量baseline...")
    baseline = measure_final_disamb(model)
    print(f"  Baseline: logit_margin={baseline['mean_logit_margin']:.4f}, "
          f"hidden_cos_dist={baseline['mean_hidden_cos_dist']:.4f}")
    
    # Sample layers to test (every 4-5 layers)
    if n_layers <= 20:
        test_layers = list(range(n_layers))
    else:
        test_layers = list(range(0, n_layers, max(1, n_layers // 8)))
        if (n_layers - 1) not in test_layers:
            test_layers.append(n_layers - 1)
    
    # Per-layer MLP ablation
    mlp_results = {}
    print(f"\n  MLP逐层消融 (测试{len(test_layers)}层)...")
    for li in test_layers:
        hooks = zero_mlp_at_layer(li, model)
        try:
            disamb = measure_final_disamb(model)
            delta_logit = disamb['mean_logit_margin'] - baseline['mean_logit_margin']
            delta_hidden = disamb['mean_hidden_cos_dist'] - baseline['mean_hidden_cos_dist']
            mlp_results[str(li)] = {
                "logit_margin": round(disamb['mean_logit_margin'], 4),
                "hidden_cos_dist": round(disamb['mean_hidden_cos_dist'], 4),
                "delta_logit": round(delta_logit, 4),
                "delta_hidden": round(delta_hidden, 4),
            }
            print(f"    L{li}: logit={disamb['mean_logit_margin']:.4f}(Δ{delta_logit:+.4f}), "
                  f"hidden={disamb['mean_hidden_cos_dist']:.4f}(Δ{delta_hidden:+.4f})")
        finally:
            remove_hooks(hooks)
        torch.cuda.empty_cache()
    
    # Per-layer Attn ablation
    attn_results = {}
    print(f"\n  Attn逐层消融 (测试{len(test_layers)}层)...")
    for li in test_layers:
        hooks = zero_attn_at_layer(li, model)
        try:
            disamb = measure_final_disamb(model)
            delta_logit = disamb['mean_logit_margin'] - baseline['mean_logit_margin']
            delta_hidden = disamb['mean_hidden_cos_dist'] - baseline['mean_hidden_cos_dist']
            attn_results[str(li)] = {
                "logit_margin": round(disamb['mean_logit_margin'], 4),
                "hidden_cos_dist": round(disamb['mean_hidden_cos_dist'], 4),
                "delta_logit": round(delta_logit, 4),
                "delta_hidden": round(delta_hidden, 4),
            }
            print(f"    L{li}: logit={disamb['mean_logit_margin']:.4f}(Δ{delta_logit:+.4f}), "
                  f"hidden={disamb['mean_hidden_cos_dist']:.4f}(Δ{delta_hidden:+.4f})")
        finally:
            remove_hooks(hooks)
        torch.cuda.empty_cache()
    
    # Find critical layers
    summary = {
        "baseline": baseline,
        "test_layers": test_layers,
        "mlp_ablation": mlp_results,
        "attn_ablation": attn_results,
    }
    
    # Most impactful MLP layer
    if mlp_results:
        mlp_impacts = {int(k): abs(v["delta_logit"]) for k, v in mlp_results.items()}
        max_mlp_li = max(mlp_impacts, key=mlp_impacts.get)
        summary["most_impactful_mlp_layer"] = max_mlp_li
        summary["most_impactful_mlp_delta"] = mlp_results[str(max_mlp_li)]["delta_logit"]
    
    # Most impactful Attn layer
    if attn_results:
        attn_impacts = {int(k): abs(v["delta_logit"]) for k, v in attn_results.items()}
        max_attn_li = max(attn_impacts, key=attn_impacts.get)
        summary["most_impactful_attn_layer"] = max_attn_li
        summary["most_impactful_attn_delta"] = attn_results[str(max_attn_li)]["delta_logit"]
    
    # MLP vs Attn total impact
    total_mlp_impact = sum(abs(v["delta_logit"]) for v in mlp_results.values()) if mlp_results else 0
    total_attn_impact = sum(abs(v["delta_logit"]) for v in attn_results.values()) if attn_results else 0
    summary["total_mlp_impact"] = round(total_mlp_impact, 4)
    summary["total_attn_impact"] = round(total_attn_impact, 4)
    
    print(f"\n  --- Stage630 摘要 ---")
    print(f"  Baseline: logit_margin={baseline['mean_logit_margin']:.4f}, hidden_cos_dist={baseline['mean_hidden_cos_dist']:.4f}")
    print(f"  最关键MLP层: L{summary.get('most_impactful_mlp_layer', 'N/A')} "
          f"(Δ={summary.get('most_impactful_mlp_delta', 'N/A')})")
    print(f"  最关键Attn层: L{summary.get('most_impactful_attn_layer', 'N/A')} "
          f"(Δ={summary.get('most_impactful_attn_delta', 'N/A')})")
    print(f"  MLP总影响: {total_mlp_impact:.4f}, Attn总影响: {total_attn_impact:.4f}")
    print(f"  MLP/Attn影响比: {total_mlp_impact/max(0.001,total_attn_impact):.2f}")
    
    return summary


# ============ Main ============

def main():
    model_key = sys.argv[1].strip().lower() if len(sys.argv) > 1 else "qwen3"
    valid_keys = ["qwen3", "deepseek7b", "glm4", "gemma4"]
    if model_key not in valid_keys:
        print(f"用法: python {Path(__file__).name} [{'|'.join(valid_keys)}]")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"Stage628-629-630: GLM4隐藏通道 + Hidden-Logit统一 + 逐层消融")
    print(f"模型: {model_key}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}")
    
    t0 = time.time()
    
    bundle = load_model_bundle(model_key)
    if isinstance(bundle, tuple):
        model, tokenizer = bundle[0], bundle[1]
    else:
        model = bundle["model"]
        tokenizer = bundle["tokenizer"]
    model.eval()
    
    try:
        # Stage628
        t1 = time.time()
        r628 = run_stage628(model, tokenizer, model_key)
        print(f"\n  Stage628完成 ({time.time()-t1:.0f}s)")
        
        torch.cuda.empty_cache()
        gc.collect()
        
        # Stage629
        t2 = time.time()
        r629 = run_stage629(model, tokenizer, model_key)
        print(f"\n  Stage629完成 ({time.time()-t2:.0f}s)")
        
        torch.cuda.empty_cache()
        gc.collect()
        
        # Stage630
        t3 = time.time()
        r630 = run_stage630(model, tokenizer, model_key)
        print(f"\n  Stage630完成 ({time.time()-t3:.0f}s)")
        
    finally:
        free_model(model)
        gc.collect()
        torch.cuda.empty_cache()
    
    total_time = time.time() - t0
    
    # Combine results
    combined = {
        "model": model_key,
        "timestamp": TIMESTAMP,
        "total_time": round(total_time, 1),
        "stage628": r628,
        "stage629": r629,
        "stage630": r630,
    }
    
    out_path = OUTPUT_DIR / f"stage628_629_630_{model_key}_{TIMESTAMP}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n结果已保存: {out_path}")
    print(f"总耗时: {total_time:.0f}s")


if __name__ == "__main__":
    main()
