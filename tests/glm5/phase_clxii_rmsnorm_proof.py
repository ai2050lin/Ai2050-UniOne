#!/usr/bin/env python3
"""
Phase CLXII: RMSNorm差分放大的严格证明
=======================================

核心目标: 证明RMSNorm下G和A反向是信息论最优的

CLXI关键发现:
  Qwen3/DS7B: GA_cos=-0.90, sign_corr<-0.80, var_red=0.03-0.09
  GLM4: GA_cos≈-0.01, sign_corr≈0, var_red>1(方差放大!)
  → RMSNorm鼓励对抗平衡, 但缺少严格证明

实验设计:
  P704: 信息论最优性证明
    - 对抗(G和A反向) vs 合作(G和A同向) vs 正交(G和A垂直)
    - 在每种配置下, 计算RMSNorm后的信息保留量
    - 如果对抗 > 合作, 则RMSNorm的差分放大是信息论最优的

  P705: 收敛性证明(梯度动力学验证)
    - 在采样层对G方向添加旋转微扰
    - 观察RMSNorm后的logit变化方向
    - 如果旋转G使其远离A(更对抗)→logit增加→梯度推动对抗

  P706: 消融实验(替换RMSNorm)
    - 推理时将RMSNorm替换为固定缩放
    - 比较替换前后G和A的对抗平衡程度

实验模型: qwen3 -> deepseek7b -> glm4 (串行, 避免OOM)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import json
import time
import argparse
import gc
from datetime import datetime
from model_utils import (
    load_model, get_layers, get_model_info,
    get_W_U, release_model, get_sample_layers
)


# ===== 通用工具函数 =====

def get_n_kv_heads(sa, model):
    """获取KV头数"""
    if hasattr(sa, 'num_key_value_heads'):
        return sa.num_key_value_heads
    elif hasattr(sa, 'config') and hasattr(sa.config, 'num_key_value_heads'):
        return sa.config.num_key_value_heads
    elif hasattr(model, 'config') and hasattr(model.config, 'num_key_value_heads'):
        return model.config.num_key_value_heads
    else:
        return get_n_heads(sa, model)


def get_n_heads(sa, model):
    """获取Q头数"""
    if hasattr(sa, 'num_heads'):
        return sa.num_heads
    elif hasattr(sa, 'config') and hasattr(sa.config, 'num_attention_heads'):
        return sa.config.num_attention_heads
    elif hasattr(model, 'config') and hasattr(model.config, 'num_attention_heads'):
        return model.config.num_attention_heads
    else:
        return 32


def get_head_dim(sa, n_q):
    """从注意力权重推导head_dim (不依赖d_model/n_heads)"""
    W_o = sa.o_proj.weight  # [d_model, n_q*head_dim]
    return W_o.shape[1] // n_q


def rms_norm(h, eps=1e-6):
    """RMSNorm: h' = h / ||h|| * sqrt(d)"""
    d = h.shape[-1]
    norm = np.sqrt(np.mean(h ** 2) + eps)
    return h * np.sqrt(d) / norm


def compute_A_contrib(sa, model, h_normed):
    """计算注意力层贡献(A项)"""
    n_kv = get_n_kv_heads(sa, model)
    n_q = get_n_heads(sa, model)
    head_dim = get_head_dim(sa, n_q)
    
    W_o = sa.o_proj.weight.detach().float().cpu().numpy()  # [d_model, n_q*head_dim]
    W_v = sa.v_proj.weight.detach().float().cpu().numpy()  # [n_kv*head_dim, d_model]
    
    V_all = W_v @ h_normed  # [n_kv*head_dim]
    
    # GQA扩展
    n_groups = n_q // n_kv
    if n_groups > 1:
        V_expanded = np.zeros(n_q * head_dim)
        for kv_h in range(n_kv):
            for g in range(n_groups):
                q_h = kv_h * n_groups + g
                V_expanded[q_h * head_dim: (q_h + 1) * head_dim] = V_all[kv_h * head_dim: (kv_h + 1) * head_dim]
        return W_o @ V_expanded
    else:
        return W_o @ V_all


def compute_G_contrib(mlp, h_normed, mlp_type):
    """计算MLP层贡献(G项)"""
    W_down = mlp.down_proj.weight.detach().float().cpu().numpy()
    if mlp_type == "split_gate_up":
        W_gate = mlp.gate_proj.weight.detach().float().cpu().numpy()
        W_up = mlp.up_proj.weight.detach().float().cpu().numpy()
        gate_logits = W_gate @ h_normed
        gate_acts = 1.0 / (1.0 + np.exp(-gate_logits))
        up_vals = W_up @ h_normed
        return W_down @ (gate_acts * up_vals)
    else:
        W_gate_up = mlp.gate_up_proj.weight.detach().float().cpu().numpy()
        half = W_gate_up.shape[0] // 2
        W_gate = W_gate_up[:half]
        W_up = W_gate_up[half:]
        gate_logits = W_gate @ h_normed
        gate_acts = 1.0 / (1.0 + np.exp(-gate_logits))
        up_vals = W_up @ h_normed
        return W_down @ (gate_acts * up_vals)


def compute_GA_contributions(layer, model, h_before, mlp_type):
    """计算单层的G和A贡献"""
    sa = layer.self_attn
    mlp = layer.mlp
    
    h_normed = rms_norm(h_before)
    A_contrib = compute_A_contrib(sa, model, h_normed)
    G_contrib = compute_G_contrib(mlp, h_normed, mlp_type)
    
    return G_contrib, A_contrib


# ============================================================
# P704: 信息论最优性证明
# ============================================================
def p704_mutual_information_optimality(model, tokenizer, device, model_info):
    """
    证明RMSNorm下G和A反向是信息论最优的
    
    方法: 在实际隐藏状态上, 人为调整G和A的方向(旋转),
    计算RMSNorm后的信息保留量
    """
    print("\n" + "="*60)
    print("P704: 信息论最优性证明")
    print("="*60)
    
    W_U = get_W_U(model)
    layers = get_layers(model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    test_words = ["apple", "cat", "the"]
    results = {}
    
    sample_layers = list(range(n_layers * 2 // 3, n_layers))
    if len(sample_layers) > 8:
        step = len(sample_layers) // 8
        sample_layers = sample_layers[::step][:8]
    
    for word in test_words:
        print(f"\n  分析词: {word}")
        word_id = tokenizer.encode(word, add_special_tokens=False)
        if len(word_id) == 0:
            continue
        word_id = word_id[0]
        W_U_word = W_U[word_id]
        W_U_word_norm = W_U_word / max(np.linalg.norm(W_U_word), 1e-10)
        
        prompt = f"The {word} is"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        layer_results = []
        
        for li in sample_layers:
            h_before = outputs.hidden_states[li][0, -1, :].float().detach().cpu().numpy()
            h_after = outputs.hidden_states[li + 1][0, -1, :].float().detach().cpu().numpy()
            
            G_contrib, A_contrib = compute_GA_contributions(
                layers[li], model, h_before, model_info.mlp_type
            )
            
            A_norm = np.linalg.norm(A_contrib)
            G_norm = np.linalg.norm(G_contrib)
            
            if A_norm < 1e-10 or G_norm < 1e-10:
                continue
            
            A_dir = A_contrib / A_norm
            G_dir = G_contrib / G_norm
            
            # 实际GA角度
            actual_cos = float(np.dot(G_dir, A_dir))
            
            # 构造正交方向 (Gram-Schmidt)
            G_dir_orth = G_dir - actual_cos * A_dir
            orth_norm = np.linalg.norm(G_dir_orth)
            if orth_norm > 1e-10:
                G_dir_orth = G_dir_orth / orth_norm
            else:
                rand_dir = np.random.randn(d_model)
                rand_dir -= np.dot(rand_dir, A_dir) * A_dir
                rn = np.linalg.norm(rand_dir)
                if rn > 1e-10:
                    G_dir_orth = rand_dir / rn
            
            # 四种配置
            configs = {
                "actual": G_dir,
                "adversarial": -A_dir,  # G = -A (完全对抗)
                "cooperative": A_dir,   # G = +A (完全合作)
                "orthogonal": G_dir_orth,  # G ⊥ A
            }
            
            for config_name, G_config_dir in configs.items():
                G_config = G_norm * G_config_dir
                h_raw = h_before + G_config + A_contrib
                h_post_rmsnorm = rms_norm(h_raw)
                
                # 信息保留: cos(h_post, W_U_word)
                info_retention = float(np.dot(h_post_rmsnorm, W_U_word_norm))
                
                # 差分信号分析
                diff_signal = G_config + A_contrib
                diff_norm = np.linalg.norm(diff_signal)
                h_raw_norm = np.linalg.norm(h_raw)
                scale_factor = np.sqrt(d_model) / max(h_raw_norm, 1e-10)
                diff_post_norm = diff_norm * scale_factor
                h_before_norm = np.linalg.norm(h_before)
                info_gain = diff_post_norm / max(h_before_norm, 1e-10)
                
                # 差分信号保留比: diff_post / diff_raw
                diff_retention = diff_post_norm / max(diff_norm, 1e-10)
                
                layer_results.append({
                    "layer": li,
                    "config": config_name,
                    "actual_GA_cos": actual_cos,
                    "info_retention": info_retention,
                    "info_gain": info_gain,
                    "diff_retention": diff_retention,
                    "scale_factor": float(scale_factor),
                })
        
        results[word] = layer_results
        
        for config in ["adversarial", "cooperative", "orthogonal", "actual"]:
            retentions = [r["info_retention"] for r in layer_results if r["config"] == config]
            gains = [r["info_gain"] for r in layer_results if r["config"] == config]
            diff_ret = [r["diff_retention"] for r in layer_results if r["config"] == config]
            if retentions:
                print(f"    {config}: retention={np.mean(retentions):.4f}, "
                      f"gain={np.mean(gains):.4f}, diff_ret={np.mean(diff_ret):.4f}")
        
        del outputs
        gc.collect()
    
    return results


# ============================================================
# P705: 收敛性证明(梯度动力学验证)
# ============================================================
def p705_convergence_proof(model, tokenizer, device, model_info):
    """
    证明梯度下降自然收敛到G和A反向
    
    方法: 对G向量添加旋转微扰, 计算logit的梯度方向
    """
    print("\n" + "="*60)
    print("P705: 收敛性证明(梯度动力学验证)")
    print("="*60)
    
    W_U = get_W_U(model)
    layers = get_layers(model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    test_words = ["apple", "cat", "the"]
    results = {}
    
    sample_layers = list(range(n_layers * 2 // 3, n_layers))
    if len(sample_layers) > 6:
        step = len(sample_layers) // 6
        sample_layers = sample_layers[::step][:6]
    
    for word in test_words:
        print(f"\n  分析词: {word}")
        word_id = tokenizer.encode(word, add_special_tokens=False)
        if len(word_id) == 0:
            continue
        word_id = word_id[0]
        W_U_word = W_U[word_id]
        
        prompt = f"The {word} is"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        layer_results = []
        
        for li in sample_layers:
            h_before = outputs.hidden_states[li][0, -1, :].float().detach().cpu().numpy()
            h_after = outputs.hidden_states[li + 1][0, -1, :].float().detach().cpu().numpy()
            
            G_contrib, A_contrib = compute_GA_contributions(
                layers[li], model, h_before, model_info.mlp_type
            )
            
            A_norm = np.linalg.norm(A_contrib)
            G_norm = np.linalg.norm(G_contrib)
            
            if A_norm < 1e-10 or G_norm < 1e-10:
                continue
            
            A_dir = A_contrib / A_norm
            G_dir = G_contrib / G_norm
            
            # 构建G的旋转平面: e1=G_dir, e2在G-A平面内垂直于G
            A_proj_G = np.dot(A_dir, G_dir) * G_dir
            A_perp_G = A_dir - A_proj_G
            A_perp_norm = np.linalg.norm(A_perp_G)
            
            if A_perp_norm < 1e-10:
                continue
            
            e1 = G_dir
            e2 = A_perp_G / A_perp_norm
            
            # 旋转角度
            angles_deg = np.linspace(-90, 90, 19)
            
            angle_results = []
            base_logit = float(W_U_word @ h_after)
            
            for angle in angles_deg:
                theta = np.radians(angle)
                G_rot_dir = np.cos(theta) * e1 + np.sin(theta) * e2
                G_rot = G_norm * G_rot_dir
                
                h_raw = h_before + G_rot + A_contrib
                h_post = rms_norm(h_raw)
                
                logit_rot = float(W_U_word @ h_post)
                logit_change = logit_rot - base_logit
                GA_cos_rot = float(np.dot(G_rot_dir, A_dir))
                
                angle_results.append({
                    "angle_deg": float(angle),
                    "GA_cos": GA_cos_rot,
                    "logit": logit_rot,
                    "logit_change": logit_change,
                })
            
            # 找到logit最大的角度
            logits = [r["logit"] for r in angle_results]
            best_idx = np.argmax(logits)
            best_angle = angle_results[best_idx]["angle_deg"]
            best_GA_cos = angle_results[best_idx]["GA_cos"]
            
            # 梯度方向: dlogit/dθ at θ=0
            # 用中心差分
            grad_at_0 = (logits[10] - logits[8]) / (2 * np.radians(10))
            
            # 如果grad_at_0 < 0: 向负角度(更对抗)旋转增加logit
            grad_direction = "adversarial" if grad_at_0 < 0 else "cooperative"
            
            layer_results.append({
                "layer": li,
                "actual_GA_cos": float(np.dot(G_dir, A_dir)),
                "best_angle": best_angle,
                "best_GA_cos": best_GA_cos,
                "grad_at_0": float(grad_at_0),
                "grad_direction": grad_direction,
            })
            
            print(f"    L{li}: GA_cos={np.dot(G_dir, A_dir):.3f}, "
                  f"best_angle={best_angle:.1f}°, "
                  f"grad_dir={grad_direction}")
        
        results[word] = layer_results
        
        adv_count = sum(1 for r in layer_results if r["grad_direction"] == "adversarial")
        total = len(layer_results)
        print(f"  → 梯度方向: adversarial={adv_count}/{total}")
        
        del outputs
        gc.collect()
    
    return results


# ============================================================
# P706: 消融实验(替换RMSNorm)
# ============================================================
def p706_ablation_rmsnorm(model, tokenizer, device, model_info):
    """
    量化RMSNorm对对抗平衡的贡献度
    
    方法: 比较RMSNorm vs 固定缩放 vs 无缩放下的对抗平衡程度
    """
    print("\n" + "="*60)
    print("P706: 消融实验(替换RMSNorm)")
    print("="*60)
    
    W_U = get_W_U(model)
    layers = get_layers(model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    test_words = ["apple", "cat", "the"]
    results = {}
    
    sample_layers = list(range(n_layers * 2 // 3, n_layers))
    if len(sample_layers) > 6:
        step = len(sample_layers) // 6
        sample_layers = sample_layers[::step][:6]
    
    for word in test_words:
        print(f"\n  分析词: {word}")
        word_id = tokenizer.encode(word, add_special_tokens=False)
        if len(word_id) == 0:
            continue
        word_id = word_id[0]
        W_U_word = W_U[word_id]
        
        prompt = f"The {word} is"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        layer_results = []
        
        for li in sample_layers:
            h_before = outputs.hidden_states[li][0, -1, :].float().detach().cpu().numpy()
            h_after = outputs.hidden_states[li + 1][0, -1, :].float().detach().cpu().numpy()
            
            G_contrib, A_contrib = compute_GA_contributions(
                layers[li], model, h_before, model_info.mlp_type
            )
            
            h_raw = h_before + G_contrib + A_contrib
            
            # 三种归一化配置
            configs = {}
            
            # 1) 原始RMSNorm
            h_rmsnorm = rms_norm(h_raw)
            configs["rmsnorm"] = {
                "logit": float(W_U_word @ h_rmsnorm),
                "G_logit": float(W_U_word @ G_contrib),
                "A_logit": float(W_U_word @ A_contrib),
            }
            
            # 2) 固定缩放: α = √d / ||h_before|| (基于输入范数)
            alpha = np.sqrt(d_model) / max(np.linalg.norm(h_before), 1e-10)
            h_fixed = h_raw * alpha
            configs["fixed_scale"] = {
                "logit": float(W_U_word @ h_fixed),
                "G_logit": float(W_U_word @ G_contrib) * alpha,
                "A_logit": float(W_U_word @ A_contrib) * alpha,
            }
            
            # 3) 无缩放
            configs["no_norm"] = {
                "logit": float(W_U_word @ h_raw),
                "G_logit": float(W_U_word @ G_contrib),
                "A_logit": float(W_U_word @ A_contrib),
            }
            
            # 4) 反向RMSNorm: 故意用||h_before+G+A||缩放(鼓励合作)
            # h_anti = h_raw * √d / ||h_raw|| (正常RMSNorm)
            # 但改为: h_anti = h_raw * ||h_raw|| / √d (放大大范数,缩小小范数)
            # 这会鼓励合作(因为合作时范数大→放大更多)
            h_anti_norm = np.sqrt(np.mean(h_raw ** 2) + 1e-6)
            anti_scale = h_anti_norm / np.sqrt(d_model)  # 反向缩放
            h_anti = h_raw * anti_scale
            configs["anti_rmsnorm"] = {
                "logit": float(W_U_word @ h_anti),
                "G_logit": float(W_U_word @ G_contrib) * anti_scale,
                "A_logit": float(W_U_word @ A_contrib) * anti_scale,
            }
            
            # 计算对抗平衡指标
            for config_name, config in configs.items():
                G_l = config["G_logit"]
                A_l = config["A_logit"]
                
                is_adversarial = (G_l * A_l < 0)
                
                sum_abs = abs(G_l + A_l)
                max_abs = max(abs(G_l), abs(A_l), 1e-10)
                opposition_ratio = sum_abs / max_abs
                
                total_abs = abs(G_l) + abs(A_l)
                variance_reduction = sum_abs / max(total_abs, 1e-10)
                
                # logit增益: 相对于h_before的贡献
                h_before_logit = float(W_U_word @ h_before)
                logit_gain = config["logit"] - h_before_logit
                
                config["is_adversarial"] = is_adversarial
                config["opposition_ratio"] = opposition_ratio
                config["variance_reduction"] = variance_reduction
                config["logit_gain"] = logit_gain
            
            layer_results.append({
                "layer": li,
                "configs": configs,
            })
            
            print(f"    L{li}: ", end="")
            for cn in ["rmsnorm", "fixed_scale", "no_norm", "anti_rmsnorm"]:
                c = configs[cn]
                print(f"{cn}: var_red={c['variance_reduction']:.3f}/gain={c['logit_gain']:.3f}  ", end="")
            print()
        
        results[word] = layer_results
        
        # 统计
        for config_name in ["rmsnorm", "fixed_scale", "no_norm", "anti_rmsnorm"]:
            var_reds = [r["configs"][config_name]["variance_reduction"] for r in layer_results]
            gains = [r["configs"][config_name]["logit_gain"] for r in layer_results]
            print(f"  {config_name}: mean_var_red={np.mean(var_reds):.3f}, "
                  f"mean_gain={np.mean(gains):.3f}")
        
        del outputs
        gc.collect()
    
    return results


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Phase CLXII: RMSNorm差分放大的严格证明")
    parser.add_argument("--model", type=str, required=True,
                       choices=["qwen3", "deepseek7b", "glm4"],
                       help="模型名称")
    args = parser.parse_args()
    
    model_name = args.model
    print(f"\n{'='*60}")
    print(f"Phase CLXII: RMSNorm差分放大的严格证明 — {model_name}")
    print(f"{'='*60}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    print(f"  模型: {model_info.model_class}, {model_info.n_layers}层, d={model_info.d_model}")
    
    t0 = time.time()
    
    p704_results = p704_mutual_information_optimality(model, tokenizer, device, model_info)
    p705_results = p705_convergence_proof(model, tokenizer, device, model_info)
    p706_results = p706_ablation_rmsnorm(model, tokenizer, device, model_info)
    
    elapsed = time.time() - t0
    print(f"\n总用时: {elapsed:.1f}s")
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "results/phase_clxii"
    os.makedirs(output_dir, exist_ok=True)
    
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(x) for x in obj]
        return obj
    
    output = {
        "phase": "CLXII",
        "model": model_name,
        "timestamp": timestamp,
        "elapsed_seconds": elapsed,
        "model_info": {
            "class": model_info.model_class,
            "n_layers": model_info.n_layers,
            "d_model": model_info.d_model,
        },
        "p704_mutual_information": convert(p704_results),
        "p705_convergence": convert(p705_results),
        "p706_ablation": convert(p706_results),
    }
    
    output_file = f"{output_dir}/phase_clxii_{model_name}_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {output_file}")
    
    release_model(model)
    
    # 打印摘要
    print(f"\n{'='*60}")
    print("Phase CLXII 摘要")
    print(f"{'='*60}")
    
    for word in ["apple", "cat", "the"]:
        if word not in p704_results:
            continue
        print(f"\n{word}:")
        
        # P704
        p704 = p704_results[word]
        for config in ["adversarial", "cooperative", "orthogonal", "actual"]:
            retentions = [r["info_retention"] for r in p704 if r["config"] == config]
            diff_rets = [r["diff_retention"] for r in p704 if r["config"] == config]
            if retentions:
                print(f"  P704 {config}: ret={np.mean(retentions):.4f}, diff_ret={np.mean(diff_rets):.4f}")
        
        # P705
        p705 = p705_results.get(word, [])
        if p705:
            adv_count = sum(1 for r in p705 if r["grad_direction"] == "adversarial")
            total = len(p705)
            print(f"  P705: gradient={adv_count}/{total} adversarial")
        
        # P706
        p706 = p706_results.get(word, [])
        if p706:
            for cn in ["rmsnorm", "fixed_scale", "no_norm", "anti_rmsnorm"]:
                vr = [r["configs"][cn]["variance_reduction"] for r in p706]
                gains = [r["configs"][cn]["logit_gain"] for r in p706]
                print(f"  P706 {cn}: var_red={np.mean(vr):.3f}, gain={np.mean(gains):.3f}")


if __name__ == "__main__":
    main()
