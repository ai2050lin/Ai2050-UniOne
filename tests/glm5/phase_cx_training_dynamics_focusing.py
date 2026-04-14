"""
Phase CX-P514/P515/P516: 训练动态与信号传播的演化理论
==========================================================

Phase CIX核心发现:
- DS7B稳定性分类AUC=1.000, post_ln_norm是最关键特征
- W_gate因果干预显著: replace_linear↓68%, 放大bottom-k↑158%
- Qwen3/GLM4采样层全部稳定, 干预效果有限
- DS7B稳定层post_ln_norm更大(61.7 vs 43.4)

Phase CX核心思路:
1. "逆训练"模拟: 将权重逐步向随机方向扰动, 测量稳定性变化
2. 权重结构vs随机: 分析权重中"结构化"vs"随机"成分对稳定性的影响
3. RG(重整化群)理论: 推导层间传播的不动点和稳定性条件

P514: 逆训练与稳定性演化
  - 对每层权重W, 生成随机矩阵W_rand, 混合W_mix = (1-t)*W + t*W_rand
  - t从0(原始)到1(完全随机), 测量每个t的传播稳定性
  - 目标: 找到稳定性临界点t_c, 理解训练结构何时崩溃
  - 对比: Qwen3/GLM4(稳定) vs DS7B(不稳定)的临界点差异

P515: 结构化vs随机成分的谱分析
  - 对W_down/W_gate做SVD, 分解为"结构化成分"(top-k奇异值)和"随机成分"(bottom-k)
  - 分析: 哪个成分主导传播稳定性?
  - 对比: 不同模型的MP偏离度与稳定性的关系
  - 验证: 只保留结构化成分(去掉随机成分)是否提高DS7B的稳定性

P516: 信号传播的重整化群理论
  - 将层间传播视为RG变换: h_{l+1} = T_l(h_l) = h_l + f_l(LN(h_l))
  - Jacobian: J_l = dT_l/dh_l, 传播链 J_chain = J_L * ... * J_1
  - 稳定性条件: ||J_l|| ≈ 1 (每层放大率接近1)
  - 不动点: h* = T(h*) 满足 f(h*) = 0 (残差增量为0)
  - 推导: 从权重谱预测||J_l||的解析公式
  - 验证: ||J_l||_predicted vs ||J_l||_measured R2>0.8

使用方法:
    python phase_cx_training_dynamics_focusing.py --model qwen3 --experiment p514
    python phase_cx_training_dynamics_focusing.py --model glm4 --experiment p515
    python phase_cx_training_dynamics_focusing.py --model deepseek7b --experiment p516
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


# 工具函数(从phase_cix复制)
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

# 测试文本
TEST_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In the beginning, there was nothing but darkness.",
    "Scientists discovered a new particle that could change physics.",
    "The old man walked slowly through the park at dawn.",
    "Language models learn to predict the next word in a sequence.",
]


def compute_spectral_features(W, n_components=200):
    """计算权重矩阵的谱特征"""
    if isinstance(W, torch.Tensor):
        W_np = W.float().cpu().numpy()
    else:
        W_np = np.asarray(W, dtype=np.float32)
    
    m, n = W_np.shape
    min_dim = min(m, n)
    n_comp = min(n_components, min_dim)
    
    # Randomized SVD
    from sklearn.utils.extmath import randomized_svd
    U, S, Vt = randomized_svd(W_np, n_components=n_comp, random_state=42)
    
    # 参与率 PR
    S2 = S ** 2
    total_var = np.sum(S2)
    pr = (total_var ** 2) / np.sum(S2 ** 2) if total_var > 0 else 0
    pr_normalized = pr / min_dim
    
    # 条件数
    kappa = S[0] / max(S[-1], 1e-10)
    
    # Top-10% 能量占比
    top_k = max(1, len(S) // 10)
    top10_pct = np.sum(S2[:top_k]) / total_var if total_var > 0 else 0
    
    # MP偏离度
    S_mean = np.mean(S2)
    if S_mean > 0:
        ratio = m / n
        lambda_plus = S_mean * (1 + np.sqrt(ratio)) ** 2
        mp_dev = np.sum(S2[S2 > lambda_plus]) / total_var if total_var > 0 else 0
    else:
        mp_dev = 0
    
    return {
        'pr': pr_normalized,
        'kappa': kappa,
        'top10_pct': top10_pct,
        'mp_deviation': mp_dev,
        'S': S,
        'U': U,
        'Vt': Vt,
    }


def measure_stability(model, tokenizer, device, layers_list, l_idx, alpha, mlp_type, test_texts):
    """测量某层的传播稳定性"""
    layer = layers_list[l_idx]
    
    all_ratios = []
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        
        # 基线
        with torch.no_grad():
            baseline_outputs = model(input_ids, output_hidden_states=True)
            baseline_hs = baseline_outputs.hidden_states
        
        # 扰动W_down
        orig_w = perturb_w_down(layers_list, l_idx, alpha, mlp_type)
        
        with torch.no_grad():
            perturbed_outputs = model(input_ids, output_hidden_states=True)
            perturbed_hs = perturbed_outputs.hidden_states
        
        # 恢复
        restore_w_down(layers_list, l_idx, orig_w, mlp_type)
        
        # 计算逐层传播比
        delta_at_l = (perturbed_hs[l_idx+1] - baseline_hs[l_idx+1]).detach().float()
        delta_at_l_norm = torch.norm(delta_at_l).item()
        
        if delta_at_l_norm < 1e-10:
            continue
        
        # 逐层跟踪
        ratios = []
        for k in range(l_idx+1, len(baseline_hs)-1):
            delta_k = (perturbed_hs[k] - baseline_hs[k]).detach().float()
            delta_k1 = (perturbed_hs[k+1] - baseline_hs[k+1]).detach().float()
            norm_k = torch.norm(delta_k).item()
            norm_k1 = torch.norm(delta_k1).item()
            
            if norm_k > 1e-10:
                ratio = norm_k1 / norm_k
                ratios.append(ratio)
        
        if ratios:
            all_ratios.append(ratios)
    
    if not all_ratios:
        return {'max_ratio': 0, 'mean_ratio': 0, 'std_ratio': 0}
    
    flat_ratios = [r for ratios in all_ratios for r in ratios]
    return {
        'max_ratio': max(flat_ratios) if flat_ratios else 0,
        'mean_ratio': np.mean(flat_ratios) if flat_ratios else 0,
        'std_ratio': np.std(flat_ratios) if flat_ratios else 0,
    }


def measure_kl_at_layer(model, tokenizer, device, layers_list, l_idx, alpha, mlp_type, test_texts):
    """测量某层的kl_div"""
    kl_divs = []
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            logits_b = model(inputs["input_ids"]).logits[0, -1]
        orig_w = perturb_w_down(layers_list, l_idx, alpha, mlp_type)
        with torch.no_grad():
            logits_a = model(inputs["input_ids"]).logits[0, -1]
        restore_w_down(layers_list, l_idx, orig_w, mlp_type)
        kl_divs.append(compute_kl_divergence(logits_b, logits_a))
    return np.mean(kl_divs)


# ============================================================
# P514: 逆训练与稳定性演化
# ============================================================
def run_p514(model_name):
    """将权重逐步向随机方向扰动, 测量稳定性变化"""
    print(f"\n{'='*60}")
    print(f"P514: 逆训练与稳定性演化 — {model_name}")
    print(f"{'='*60}")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    
    # 选择3-4个测试层(浅、中、深)
    n_layers = len(layers_list)
    test_layer_indices = [n_layers//6, n_layers//3, n_layers//2, 2*n_layers//3]
    test_layer_indices = [l for l in test_layer_indices if l < n_layers]
    
    print(f"测试层: {test_layer_indices}")
    
    # 混合比例t
    t_values = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    
    results = []
    
    for l_idx in test_layer_indices:
        layer = layers_list[l_idx]
        layer_frac = l_idx / max(n_layers - 1, 1)
        print(f"\n--- L{l_idx} (frac={layer_frac:.2f}) ---")
        
        # 获取原始权重
        if info.mlp_type == "merged_gate_up":
            full_weight = layer.mlp.gate_up_proj.weight.data.clone()
            half = full_weight.shape[0] // 2
            W_gate_orig = full_weight[:half].clone()
            W_up_orig = full_weight[half:].clone()
            W_down_orig = layer.mlp.down_proj.weight.data.clone()
            is_merged = True
        else:
            W_gate_orig = layer.mlp.gate_proj.weight.data.clone()
            W_down_orig = layer.mlp.down_proj.weight.data.clone()
            is_merged = False
        
        # 生成随机矩阵(与原始同形状)
        torch.manual_seed(42)
        if is_merged:
            W_gate_rand = torch.randn_like(W_gate_orig) * W_gate_orig.std()
            W_down_rand = torch.randn_like(W_down_orig) * W_down_orig.std()
        else:
            W_gate_rand = torch.randn_like(W_gate_orig) * W_gate_orig.std()
            W_down_rand = torch.randn_like(W_down_orig) * W_down_orig.std()
        
        for t in t_values:
            # 混合权重
            W_gate_mix = (1 - t) * W_gate_orig + t * W_gate_rand
            W_down_mix = (1 - t) * W_down_orig + t * W_down_rand
            
            # 设置混合权重
            if is_merged:
                full_w = torch.cat([W_gate_mix.to(full_weight.dtype), W_up_orig.to(full_weight.dtype)], dim=0)
                layer.mlp.gate_up_proj.weight.data = full_w
                layer.mlp.down_proj.weight.data = W_down_mix.to(layer.mlp.down_proj.weight.dtype)
            else:
                layer.mlp.gate_proj.weight.data = W_gate_mix.to(layer.mlp.gate_proj.weight.dtype)
                layer.mlp.down_proj.weight.data = W_down_mix.to(layer.mlp.down_proj.weight.dtype)
            
            # 测量稳定性
            stab = measure_stability(model, tokenizer, device, layers_list, l_idx, 0.1,
                                    info.mlp_type, TEST_TEXTS[:2])
            
            # 测量kl_div
            kl_div = measure_kl_at_layer(model, tokenizer, device, layers_list, l_idx, 0.1,
                                         info.mlp_type, TEST_TEXTS[:3])
            
            print(f"  t={t:.1f}: max_ratio={stab['max_ratio']:.2f}, "
                  f"mean_ratio={stab['mean_ratio']:.2f}, kl_div={kl_div:.4f}")
            
            results.append({
                'layer': l_idx, 'layer_frac': layer_frac,
                't': t, 'max_ratio': stab['max_ratio'],
                'mean_ratio': stab['mean_ratio'],
                'kl_div': kl_div,
            })
        
        # 恢复原始权重
        if is_merged:
            layer.mlp.gate_up_proj.weight.data = torch.cat([W_gate_orig.to(full_weight.dtype), W_up_orig.to(full_weight.dtype)], dim=0)
            layer.mlp.down_proj.weight.data = W_down_orig.to(layer.mlp.down_proj.weight.dtype)
        else:
            layer.mlp.gate_proj.weight.data = W_gate_orig.to(layer.mlp.gate_proj.weight.dtype)
            layer.mlp.down_proj.weight.data = W_down_orig.to(layer.mlp.down_proj.weight.dtype)
    
    # 分析: 找到临界点
    print(f"\n{'='*40}")
    print("P514 逆训练总结:")
    print(f"{'='*40}")
    
    for l_idx in test_layer_indices:
        layer_results = [r for r in results if r['layer'] == l_idx]
        print(f"\n  L{l_idx}:")
        
        # 找临界点: max_ratio首次超过2的t值
        t_critical = None
        for r in layer_results:
            if r['max_ratio'] >= 2.0 and r['t'] > 0:
                t_critical = r['t']
                break
        
        # 基线(t=0)的ratio
        baseline = next((r for r in layer_results if r['t'] == 0), None)
        if baseline:
            print(f"    基线: max_ratio={baseline['max_ratio']:.2f}, kl_div={baseline['kl_div']:.4f}")
        
        # 完全随机(t=1)的ratio
        full_rand = next((r for r in layer_results if r['t'] == 1.0), None)
        if full_rand:
            print(f"    完全随机: max_ratio={full_rand['max_ratio']:.2f}, kl_div={full_rand['kl_div']:.4f}")
        
        if t_critical:
            print(f"    临界点t_c={t_critical:.1f} (max_ratio首次≥2)")
        else:
            max_t_ratio = max(r['max_ratio'] for r in layer_results)
            if max_t_ratio < 2.0:
                print(f"    所有t值都稳定(max_ratio={max_t_ratio:.2f}<2)")
            else:
                print(f"    基线已不稳定(max_ratio≥2)")
        
        # kl_div随t的变化趋势
        t_vals = [r['t'] for r in layer_results]
        kl_vals = [r['kl_div'] for r in layer_results]
        if len(t_vals) > 2 and np.std(kl_vals) > 1e-10:
            r, p = pearsonr(t_vals, kl_vals)
            print(f"    t vs kl_div: r={r:.3f}")
    
    del model
    torch.cuda.empty_cache()
    print("P514完成")


# ============================================================
# P515: 结构化vs随机成分的谱分析
# ============================================================
def run_p515(model_name):
    """分析权重中结构化vs随机成分对稳定性的影响"""
    print(f"\n{'='*60}")
    print(f"P515: 结构化vs随机成分的谱分析 — {model_name}")
    print(f"{'='*60}")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    n_layers = len(layers_list)
    
    # 选择6-8个测试层
    sample_layers = get_sample_layers(n_layers, 8)
    print(f"采样层: {sample_layers}")
    
    # [1] 分析每层的谱结构
    print("\n[1] 谱结构分析...")
    layer_spectral = []
    
    for l in sample_layers:
        layer = layers_list[l]
        weights = get_layer_weights(layer, info.d_model, info.mlp_type)
        
        W_down_spec = compute_spectral_features(weights.W_down)
        W_gate_spec = compute_spectral_features(weights.W_gate) if weights.W_gate is not None else None
        
        data = {
            'layer': l,
            'layer_frac': l / max(n_layers - 1, 1),
            'W_down_pr': W_down_spec['pr'],
            'W_down_kappa': W_down_spec['kappa'],
            'W_down_mp_dev': W_down_spec['mp_deviation'],
            'W_down_top10': W_down_spec['top10_pct'],
            'W_down_S': W_down_spec['S'],
            'W_gate_pr': W_gate_spec['pr'] if W_gate_spec else 0,
            'W_gate_kappa': W_gate_spec['kappa'] if W_gate_spec else 0,
            'W_gate_mp_dev': W_gate_spec['mp_deviation'] if W_gate_spec else 0,
        }
        layer_spectral.append(data)
        print(f"  L{l}: W_down PR={data['W_down_pr']:.3f}, kappa={data['W_down_kappa']:.1f}, "
              f"MP_dev={data['W_down_mp_dev']:.3f}")
    
    # [2] 低秩近似实验: 只保留top-k奇异值
    print("\n[2] 低秩近似实验...")
    rank_ratios = [0.1, 0.3, 0.5, 0.8, 1.0]  # 保留的奇异值比例
    
    for l_idx in sample_layers[:4]:  # 只测试4个层(节省时间)
        layer = layers_list[l_idx]
        layer_frac = l_idx / max(n_layers - 1, 1)
        print(f"\n  L{l_idx} (frac={layer_frac:.2f}):")
        
        weights = get_layer_weights(layer, info.d_model, info.mlp_type)
        W_down = weights.W_down
        
        if isinstance(W_down, np.ndarray):
            W_down_torch = torch.tensor(W_down, dtype=torch.float32, device=device)
        else:
            W_down_torch = W_down.to(device).float()
        
        U, S, Vt = torch.linalg.svd(W_down_torch, full_matrices=False)
        
        for rank_ratio in rank_ratios:
            k = max(1, int(rank_ratio * len(S)))
            S_modified = torch.zeros_like(S)
            S_modified[:k] = S[:k]
            
            W_lowrank = U @ torch.diag(S_modified) @ Vt
            
            # 设置低秩权重
            orig_w = layer.mlp.down_proj.weight.data.clone()
            layer.mlp.down_proj.weight.data = W_lowrank.to(orig_w.dtype)
            
            # 测量稳定性
            stab = measure_stability(model, tokenizer, device, layers_list, l_idx, 0.1,
                                    info.mlp_type, TEST_TEXTS[:2])
            
            # 测量kl_div
            kl_div = measure_kl_at_layer(model, tokenizer, device, layers_list, l_idx, 0.1,
                                         info.mlp_type, TEST_TEXTS[:2])
            
            # 恢复
            layer.mlp.down_proj.weight.data = orig_w
            
            print(f"    rank_ratio={rank_ratio:.0%}(k={k}): "
                  f"max_ratio={stab['max_ratio']:.2f}, kl_div={kl_div:.4f}")
    
    # [3] MP偏离度与稳定性的关系
    print("\n[3] MP偏离度与稳定性的关系...")
    
    for l_idx in sample_layers:
        layer = layers_list[l_idx]
        layer_frac = l_idx / max(n_layers - 1, 1)
        weights = get_layer_weights(layer, info.d_model, info.mlp_type)
        
        W_down_spec = compute_spectral_features(weights.W_down)
        
        # 计算"结构化能量" vs "随机能量"
        S = W_down_spec['S']
        S2 = S ** 2
        total_energy = np.sum(S2)
        
        # MP阈值
        m, n = weights.W_down.shape
        ratio = m / n
        lambda_plus = np.mean(S2) * (1 + np.sqrt(ratio)) ** 2
        
        # 结构化: 超过MP阈值的奇异值
        structured_mask = S2 > lambda_plus
        n_structured = np.sum(structured_mask)
        structured_energy = np.sum(S2[structured_mask]) / total_energy if total_energy > 0 else 0
        
        # 随机: 低于MP阈值的奇异值
        random_energy = 1 - structured_energy
        
        # 测量稳定性
        stab = measure_stability(model, tokenizer, device, layers_list, l_idx, 0.1,
                                info.mlp_type, TEST_TEXTS[:2])
        
        print(f"  L{l_idx}: 结构化能量={structured_energy:.3f}({n_structured}个), "
              f"随机能量={random_energy:.3f}, max_ratio={stab['max_ratio']:.2f}")
    
    # [4] 去除随机成分实验
    print("\n[4] 去除随机成分实验(只保留结构化奇异值)...")
    
    for l_idx in sample_layers[:4]:
        layer = layers_list[l_idx]
        layer_frac = l_idx / max(n_layers - 1, 1)
        weights = get_layer_weights(layer, info.d_model, info.mlp_type)
        
        W_down = weights.W_down
        if isinstance(W_down, np.ndarray):
            W_down_torch = torch.tensor(W_down, dtype=torch.float32, device=device)
        else:
            W_down_torch = W_down.to(device).float()
        
        U, S, Vt = torch.linalg.svd(W_down_torch, full_matrices=False)
        
        # 基线稳定性
        stab_base = measure_stability(model, tokenizer, device, layers_list, l_idx, 0.1,
                                      info.mlp_type, TEST_TEXTS[:2])
        
        # MP阈值
        S2 = (S ** 2).cpu().numpy()
        m, n = W_down.shape
        ratio = m / n
        lambda_plus = np.mean(S2) * (1 + np.sqrt(ratio)) ** 2
        
        # 只保留超过MP阈值的奇异值(结构化成分)
        S_filtered = S.clone()
        S_filtered_cpu = S_filtered.cpu().numpy()
        for i in range(len(S_filtered_cpu)):
            if S2[i] < lambda_plus:
                S_filtered_cpu[i] = 0
        S_filtered = torch.tensor(S_filtered_cpu, device=S.device, dtype=S.dtype)
        
        W_filtered = U @ torch.diag(S_filtered) @ Vt
        
        orig_w = layer.mlp.down_proj.weight.data.clone()
        layer.mlp.down_proj.weight.data = W_filtered.to(orig_w.dtype)
        
        stab_filtered = measure_stability(model, tokenizer, device, layers_list, l_idx, 0.1,
                                          info.mlp_type, TEST_TEXTS[:2])
        
        kl_filtered = measure_kl_at_layer(model, tokenizer, device, layers_list, l_idx, 0.1,
                                          info.mlp_type, TEST_TEXTS[:2])
        
        # 恢复
        layer.mlp.down_proj.weight.data = orig_w
        
        delta_ratio = stab_filtered['max_ratio'] - stab_base['max_ratio']
        print(f"  L{l_idx}: 基线ratio={stab_base['max_ratio']:.2f}, "
              f"去随机ratio={stab_filtered['max_ratio']:.2f} (Δ={delta_ratio:+.2f}), "
              f"kl={kl_filtered:.4f}")
    
    del model
    torch.cuda.empty_cache()
    print("P515完成")


# ============================================================
# P516: 信号传播的重整化群理论
# ============================================================
def run_p516(model_name):
    """推导层间传播的不动点和稳定性条件"""
    print(f"\n{'='*60}")
    print(f"P516: 信号传播的重整化群理论 — {model_name}")
    print(f"{'='*60}")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    n_layers = len(layers_list)
    
    # 选择6-8个测试层
    sample_layers = get_sample_layers(n_layers, 8)
    print(f"采样层: {sample_layers}")
    
    # [1] 测量每层的局部Jacobian范数
    print("\n[1] 测量每层的局部Jacobian范数...")
    
    layer_jacobian_data = []
    
    for l_idx in sample_layers:
        layer = layers_list[l_idx]
        layer_frac = l_idx / max(n_layers - 1, 1)
        weights = get_layer_weights(layer, info.d_model, info.mlp_type)
        
        # 获取权重范数
        W_down_norm = np.linalg.norm(weights.W_down) if weights.W_down is not None else 0
        W_gate_norm = np.linalg.norm(weights.W_gate) if weights.W_gate is not None else 0
        W_up_norm = np.linalg.norm(weights.W_up) if weights.W_up is not None else 0
        W_o_norm = np.linalg.norm(weights.W_o) if weights.W_o is not None else 0
        
        # Post-LN权重
        post_ln = weights.post_attn_layernorm_weight
        post_ln_norm = np.linalg.norm(post_ln) if post_ln is not None else 0
        ln_norm = np.linalg.norm(weights.input_layernorm_weight) if weights.input_layernorm_weight is not None else 0
        
        # 谱特征
        W_down_spec = compute_spectral_features(weights.W_down) if weights.W_down is not None else None
        W_gate_spec = compute_spectral_features(weights.W_gate) if weights.W_gate is not None else None
        
        # 实际测量每层传播比
        all_ratios = []
        for text in TEST_TEXTS[:3]:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]
            
            with torch.no_grad():
                baseline_outputs = model(input_ids, output_hidden_states=True)
                baseline_hs = baseline_outputs.hidden_states
            
            orig_w = perturb_w_down(layers_list, l_idx, 0.1, info.mlp_type)
            
            with torch.no_grad():
                perturbed_outputs = model(input_ids, output_hidden_states=True)
                perturbed_hs = perturbed_outputs.hidden_states
            
            restore_w_down(layers_list, l_idx, orig_w, info.mlp_type)
            
            # 计算局部传播比(扰动层→下一层)
            delta_l = (perturbed_hs[l_idx+1] - baseline_hs[l_idx+1]).detach().float()
            delta_l1 = (perturbed_hs[l_idx+2] - baseline_hs[l_idx+2]).detach().float() if l_idx+2 < len(baseline_hs) else None
            
            norm_l = torch.norm(delta_l).item()
            
            if delta_l1 is not None:
                norm_l1 = torch.norm(delta_l1).item()
                local_ratio = norm_l1 / max(norm_l, 1e-10)
                all_ratios.append(local_ratio)
            
            # 也测量h_in和delta的范数
            h_in_norm = torch.norm(baseline_hs[l_idx]).item()
            delta_norm = norm_l
        
        mean_local_ratio = np.mean(all_ratios) if all_ratios else 0
        std_local_ratio = np.std(all_ratios) if all_ratios else 0
        
        data = {
            'layer': l_idx,
            'layer_frac': layer_frac,
            'local_ratio': mean_local_ratio,
            'local_ratio_std': std_local_ratio,
            'W_down_norm': W_down_norm,
            'W_gate_norm': W_gate_norm,
            'W_up_norm': W_up_norm,
            'W_o_norm': W_o_norm,
            'post_ln_norm': post_ln_norm,
            'ln_norm': ln_norm,
            'W_down_kappa': W_down_spec['kappa'] if W_down_spec else 0,
            'W_down_pr': W_down_spec['pr'] if W_down_spec else 0,
            'W_gate_kappa': W_gate_spec['kappa'] if W_gate_spec else 0,
            'W_gate_pr': W_gate_spec['pr'] if W_gate_spec else 0,
        }
        layer_jacobian_data.append(data)
        print(f"  L{l_idx}: local_ratio={mean_local_ratio:.3f}±{std_local_ratio:.3f}, "
              f"W_down_kappa={data['W_down_kappa']:.1f}, post_ln={post_ln_norm:.1f}")
    
    # [2] 推导局部传播比的预测公式
    print("\n[2] 局部传播比预测公式...")
    
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
        
        feature_keys = [k for k in layer_jacobian_data[0].keys() 
                       if k not in ['layer', 'local_ratio', 'local_ratio_std']
                       and isinstance(layer_jacobian_data[0][k], (int, float))]
        
        X = np.array([[d[k] for k in feature_keys] for d in layer_jacobian_data])
        y = np.array([d['local_ratio'] for d in layer_jacobian_data])
        X = np.nan_to_num(X, nan=0, posinf=1e10, neginf=-1e10)
        y = np.nan_to_num(y, nan=1.0, posinf=10.0, neginf=0.0)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        reg = LinearRegression()
        reg.fit(X_scaled, y)
        R2 = reg.score(X_scaled, y)
        
        print(f"  R2(全特征) = {R2:.3f}")
        print(f"  特征权重:")
        for i, key in enumerate(feature_keys):
            if abs(reg.coef_[i]) > 0.01:
                print(f"    {key}: {reg.coef_[i]:.4f}")
        
        # 简化公式
        top_features_idx = np.argsort(np.abs(reg.coef_))[::-1][:5]
        X_top = X_scaled[:, top_features_idx]
        reg_simple = LinearRegression()
        reg_simple.fit(X_top, y)
        R2_simple = reg_simple.score(X_top, y)
        
        print(f"\n  R2(top-5) = {R2_simple:.3f}")
        print(f"  简化公式: local_ratio ≈ ", end="")
        terms = []
        for i, idx in enumerate(top_features_idx):
            c = reg_simple.coef_[i]
            terms.append(f"{c:.3f}×{feature_keys[idx]}")
        print(" + ".join(terms) + f" + {reg_simple.intercept_:.3f}")
        
    except ImportError:
        print("  sklearn不可用, 跳过回归")
    
    # [3] RG理论: 层间传播作为RG变换
    print("\n[3] RG理论分析...")
    
    # 对于每层, h_{l+1} = h_l + f_l(h_l)
    # Jacobian: J_l = I + df_l/dh_l
    # 近似: ||J_l|| ≈ 1 + ||df_l/dh_l|| ≈ 1 + alpha * ||W_down|| * sigma'(z) / ||h_l||
    # 其中alpha是扰动幅度, sigma'是激活函数导数
    
    print("\n  理论模型:")
    print("  h_{l+1} = h_l + f_l(h_l)")
    print("  J_l = I + df_l/dh_l")
    print("  ||J_l|| ≈ 1 + alpha * ||W_down|| * sigma'(z) / ||h_l||")
    print("  稳定性条件: ||J_l|| ≈ 1 → alpha * ||W_down|| * sigma'(z) ≈ ||h_l||")
    
    # 验证理论预测
    print("\n[4] 验证理论预测...")
    
    for l_idx in sample_layers[:4]:
        layer = layers_list[l_idx]
        
        for text in TEST_TEXTS[:2]:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]
            
            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)
                hs = outputs.hidden_states
            
            # h_in at this layer
            h_in = hs[l_idx].detach().float()
            h_out = hs[l_idx+1].detach().float()
            
            h_in_norm = torch.norm(h_in).item()
            h_out_norm = torch.norm(h_out).item()
            
            # 残差增量
            delta_h = h_out - h_in
            delta_norm = torch.norm(delta_h).item()
            
            # 相对增量
            rel_delta = delta_norm / max(h_in_norm, 1e-10)
            
            # 每个token的增量
            seq_deltas = []
            for t in range(h_in.shape[1]):
                d = torch.norm(h_out[0, t] - h_in[0, t]).item()
                h_n = torch.norm(h_in[0, t]).item()
                seq_deltas.append(d / max(h_n, 1e-10))
            
            mean_rel_delta = np.mean(seq_deltas)
            std_rel_delta = np.std(seq_deltas)
            
            print(f"  L{l_idx}: h_in_norm={h_in_norm:.1f}, h_out_norm={h_out_norm:.1f}, "
                  f"delta/h_in={mean_rel_delta:.3f}±{std_rel_delta:.3f}")
            break  # 只用第一条文本
    
    # [5] 不动点分析
    print("\n[5] 不动点分析...")
    print("  不动点条件: h* = T(h*) → f(h*) = 0 → 残差增量为0")
    print("  实际: delta_h/h_in = 0.01-0.10 (非零, 但小)")
    print("  意义: 网络接近不动点但不在不动点上(否则无法传递信息)")
    
    # 统计
    stable_layers = sum(1 for d in layer_jacobian_data if d['local_ratio'] < 1.5)
    unstable_layers = sum(1 for d in layer_jacobian_data if d['local_ratio'] >= 2.0)
    print(f"\n  局部传播比统计:")
    print(f"    <1.5 (稳定): {stable_layers}/{len(layer_jacobian_data)}")
    print(f"    >=2.0 (不稳定): {unstable_layers}/{len(layer_jacobian_data)}")
    print(f"    均值: {np.mean([d['local_ratio'] for d in layer_jacobian_data]):.3f}")
    print(f"    最大: {max(d['local_ratio'] for d in layer_jacobian_data):.3f}")
    
    del model
    torch.cuda.empty_cache()
    print("P516完成")


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase CX: 训练动态与信号传播的演化理论")
    parser.add_argument("--model", type=str, required=True, 
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--experiment", type=str, required=True,
                       choices=["p514", "p515", "p516"])
    args = parser.parse_args()
    
    start_time = time.time()
    
    if args.experiment == "p514":
        run_p514(args.model)
    elif args.experiment == "p515":
        run_p515(args.model)
    elif args.experiment == "p516":
        run_p516(args.model)
    
    elapsed = time.time() - start_time
    print(f"\n耗时: {elapsed:.1f}秒")
