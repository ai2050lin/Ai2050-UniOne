#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage619-620-621: SwiGLU维度压缩精确方程 + Gate-Up交互分析 + MLP分段线性近似

Stage619: SwiGLU稀疏激活分析（P0优先级）
  原理：Stage618发现权重满秩(>1000)但变换低维(1.6-5.2)，SwiGLU是维度压缩桥梁。
  - 从权重矩阵解析计算gate激活值: g = act_fn(x @ W_gate)
  - 测量稀疏度：gate > threshold的神经元比例
  - 测量活跃神经元一致性：哪些神经元跨输入始终活跃
  - 计算有效维度与稀疏度的关系

Stage620: Gate-Up逐元素交互分析
  原理：理解gate和up之间的数学关系。
  - 计算gate和up激活的逐元素相关系数
  - 检验gate是否是简单阈值门控（gate > 0.5 → 通过，否则 → 0）
  - 测量"门控效率"：gate值分布与最终输出能量的关系
  - 检验不同阈值下的有效维度变化

Stage621: MLP变换的分段线性近似
  原理：SwiGLU的非线性来自激活函数，在gate值的不同区间可能有不同行为。
  - 按gate值将neuron分为"活跃"(>0.5)、"半活跃"(0.1-0.5)、"沉默"(<0.1)三组
  - 在每组内拟合线性变换矩阵
  - 检查分段变换的结构：活跃组是否构成低秩子空间
  - 验证"稀疏激活→低维变换"的因果链

用法: python stage619_620_621_swiglu_compression.py [qwen3|deepseek7b|glm4|gemma4]
"""

from __future__ import annotations
import sys, json, time, gc, torch, os
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


def find_components(layer_module):
    attn_mod = None
    mlp_mod = None
    for name, child in layer_module.named_children():
        nl = name.lower()
        if 'attn' in nl or 'attention' in nl or 'self_attn' in nl:
            attn_mod = child
        if 'mlp' in nl or 'feed_forward' in nl or 'ffn' in nl:
            mlp_mod = child
    return attn_mod, mlp_mod


def get_activation_fn(model):
    """检测模型的激活函数类型"""
    config = getattr(model, 'config', None) or getattr(getattr(model, 'model', None), 'config', None)
    if config is None:
        return 'silu'

    hidden_act = getattr(config, 'hidden_act', None)
    if hidden_act:
        return str(hidden_act).lower()
    return 'silu'


def apply_activation(x, act_name):
    """应用对应的激活函数"""
    if 'gelu' in act_name:
        return F.gelu(x)
    elif 'relu' in act_name:
        return F.relu(x)
    elif 'silu' in act_name or 'swish' in act_name:
        return F.silu(x)
    else:
        return F.silu(x)  # 默认SiLU


def get_mlp_weights(mlp_mod, act_name):
    """
    从MLP模块中提取权重矩阵。
    返回: (W_gate, W_up, W_down, is_merged)
    - is_merged=True表示gate和up合并（如GLM4）
    """
    W_gate = None
    W_up = None
    W_down = None
    is_merged = False

    # 递归查找所有权重
    all_params = {}
    for name, param in mlp_mod.named_parameters():
        if param.dim() >= 2:
            all_params[name] = param

    # 检测合并的gate_up_proj
    gate_keys = [k for k in all_params.keys() if 'gate' in k.lower()]
    up_keys = [k for k in all_params.keys() if 'up' in k.lower()]
    down_keys = [k for k in all_params.keys() if 'down' in k.lower()]

    # GLM4风格: gate_up_proj合并
    merged_keys = [k for k in all_params.keys() if 'gate_up' in k.lower()]
    if merged_keys:
        merged_w = all_params[merged_keys[0]].float().detach().cpu()
        # 合并矩阵通常是 [2*intermediate, hidden]，拆分为gate和up
        half = merged_w.shape[0] // 2
        W_gate = merged_w[:half, :]  # [intermediate, hidden]
        W_up = merged_w[half:, :]    # [intermediate, hidden]
        is_merged = True
    elif gate_keys and up_keys:
        W_gate = all_params[gate_keys[0]].float().detach().cpu()
        W_up = all_params[up_keys[0]].float().detach().cpu()
    else:
        # DS7B或其他MoE架构：只有一个权重
        # 找最大的非down权重
        for name, param in all_params.items():
            if 'down' not in name.lower():
                if W_gate is None or param.shape[0] > W_gate.shape[0]:
                    W_gate = param.float().detach().cpu()

    if down_keys:
        W_down = all_params[down_keys[0]].float().detach().cpu()

    return W_gate, W_up, W_down, is_merged


PROBE_SENTENCES = [
    "The cat sat on the mat.",
    "A beautiful sunset over the ocean.",
    "The quantum physics lecture was fascinating.",
    "She traveled to Paris last summer.",
    "The company reported strong earnings.",
    "He played guitar in the band.",
    "The recipe calls for fresh ingredients.",
    "Climate change affects global weather patterns.",
    "The museum exhibits ancient artifacts.",
    "Digital technology transforms education.",
    "The old bridge crossed the river.",
    "Music brings joy to everyone.",
]


# ============ Stage619: SwiGLU稀疏激活分析 ============

def run_stage619(model, tokenizer, model_key):
    """
    分析SwiGLU的稀疏激活模式。

    原理：
    1. 从权重矩阵W_gate和W_up解析计算gate激活值 g = act_fn(x @ W_gate)
    2. 统计不同阈值下的活跃神经元比例
    3. 分析活跃神经元跨输入的一致性
    4. 计算有效维度与稀疏度的关系
    """
    print(f"\n  --- Stage619: SwiGLU稀疏激活分析 ---")
    t0 = time.time()
    device = safe_get_device(model)
    layers = discover_layers(model)
    n_layers = len(layers)
    act_name = get_activation_fn(model)
    print(f"  激活函数: {act_name}")

    hidden_dim = None
    intermediate_dim = None

    # Sample layers
    sample_layers = list(range(0, n_layers, max(1, n_layers // 8)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)

    # Collect MLP inputs for all probe sentences
    layer_inputs = {}  # layer_idx -> list of input vectors

    for sent in PROBE_SENTENCES:
        enc = tokenizer(sent, return_tensors="pt", truncation=True, max_length=128)
        enc = move_to_device(enc, model)

        local_inputs = {}
        hooks = []

        for li in sample_layers:
            layer = layers[li]
            _, mlp_mod = find_components(layer)
            if mlp_mod is None:
                continue

            if hidden_dim is None:
                W_gate, _, _, _ = get_mlp_weights(mlp_mod, act_name)
                if W_gate is not None:
                    hidden_dim = W_gate.shape[1]
                    intermediate_dim = W_gate.shape[0]

            def make_pre_hook(layer_idx):
                def pre_hook(module, input):
                    if isinstance(input, tuple) and len(input) > 0:
                        x = input[0]
                        if x.dim() >= 2:
                            local_inputs[layer_idx] = x[0, -1, :].float().detach().cpu()
                return pre_hook

            hooks.append(mlp_mod.register_forward_pre_hook(make_pre_hook(li)))

        try:
            with torch.no_grad():
                model(**enc)
        finally:
            for h in hooks:
                h.remove()

        for li in sample_layers:
            if li in local_inputs:
                layer_inputs.setdefault(li, []).append(local_inputs[li])

    # Analyze sparsity for each layer
    results = {}
    for li in sample_layers:
        if li not in layer_inputs or len(layer_inputs[li]) < 3:
            continue

        layer = layers[li]
        _, mlp_mod = find_components(layer)
        if mlp_mod is None:
            continue

        W_gate, W_up, W_down, is_merged = get_mlp_weights(mlp_mod, act_name)
        if W_gate is None or intermediate_dim is None:
            continue

        inputs = torch.stack(layer_inputs[li], dim=0)  # [N, hidden]
        N = inputs.shape[0]

        # Compute gate activations: g = act_fn(x @ W_gate^T) -> [N, intermediate]
        gate_pre = inputs @ W_gate.T  # [N, intermediate]
        gate_act = apply_activation(gate_pre, act_name)  # [N, intermediate]

        # Compute up activations: u = x @ W_up^T -> [N, intermediate]
        if W_up is not None:
            up_act = inputs @ W_up.T  # [N, intermediate]
        else:
            up_act = gate_pre  # 如果gate和up合并

        # Gated output before down_proj: h = gate * up -> [N, intermediate]
        gated = gate_act * up_act

        # 1. Sparsity analysis at different thresholds
        thresholds = [0.05, 0.1, 0.2, 0.3, 0.5]
        sparsity = {}
        for thresh in thresholds:
            active_fraction = (gate_act > thresh).float().mean().item()
            sparsity[f"active_{thresh}"] = active_fraction

        # 2. Gate statistics
        gate_mean = gate_act.mean().item()
        gate_std = gate_act.std().item()
        gate_max = gate_act.max().item()
        gate_median = gate_act.median().item()

        # 3. Neuron consistency: which neurons are active across most inputs?
        active_per_neuron = (gate_act > 0.3).float().mean(dim=0)  # [intermediate]
        always_active = (active_per_neuron > 0.8).sum().item()
        never_active = (active_per_neuron < 0.2).sum().item()
        sometimes_active = intermediate_dim - always_active - never_active

        # 4. Effective dimension of gated output (rank analysis)
        # Normalize gated outputs
        gated_centered = gated - gated.mean(dim=0, keepdim=True)
        if N >= 3:
            sv = torch.linalg.svdvals(gated_centered)
            total_energy = (sv ** 2).sum().item()
            if total_energy > 1e-10:
                cum_energy = torch.cumsum(sv ** 2, dim=0) / total_energy
                eff_dim_90 = (cum_energy < 0.90).sum().item() + 1
                eff_dim_95 = (cum_energy < 0.95).sum().item() + 1
                eff_dim_99 = (cum_energy < 0.99).sum().item() + 1
            else:
                eff_dim_90 = eff_dim_95 = eff_dim_99 = 0
        else:
            eff_dim_90 = eff_dim_95 = eff_dim_99 = N

        # 5. Energy concentration of top-k singular values
        if N >= 3:
            top1_energy = (sv[0] ** 2 / total_energy).item() if total_energy > 1e-10 else 0
            top5_energy = (sv[:5] ** 2).sum().item() / total_energy if total_energy > 1e-10 and len(sv) >= 5 else 0
        else:
            top1_energy = top5_energy = 0

        # 6. Gate value distribution shape (kurtosis-like metric)
        gate_flat = gate_act.flatten().numpy()
        gate_q25 = np.percentile(gate_flat, 25)
        gate_q75 = np.percentile(gate_flat, 75)
        gate_iqr = gate_q75 - gate_q25

        layer_data = {
            "intermediate_dim": intermediate_dim,
            "hidden_dim": hidden_dim,
            "is_merged": is_merged,
            "gate_mean": round(gate_mean, 4),
            "gate_std": round(gate_std, 4),
            "gate_max": round(gate_max, 4),
            "gate_median": round(gate_median, 4),
            "gate_q25": round(float(gate_q25), 4),
            "gate_q75": round(float(gate_q75), 4),
            "gate_iqr": round(float(gate_iqr), 4),
            "always_active": always_active,
            "never_active": never_active,
            "sometimes_active": sometimes_active,
            "eff_dim_90": eff_dim_90,
            "eff_dim_95": eff_dim_95,
            "eff_dim_99": eff_dim_99,
            "top1_energy": round(top1_energy, 4),
            "top5_energy": round(top5_energy, 4),
        }
        for k, v in sparsity.items():
            layer_data[k] = round(v, 4)

        results[str(li)] = layer_data
        print(f"    L{li}: gate_mean={gate_mean:.3f}, gate_std={gate_std:.3f}, "
              f"active_0.3={sparsity['active_0.3']:.3f}, "
              f"always_active={always_active}/{intermediate_dim}, "
              f"eff_dim_90={eff_dim_90}")

        # Free memory
        del gate_pre, gate_act, up_act, gated, gated_centered, sv
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"  Stage619完成，耗时{elapsed:.1f}s，分析{len(results)}层")

    # Aggregate stats
    if results:
        active_03_vals = [v['active_0.3'] for v in results.values()]
        eff_dim_vals = [v['eff_dim_90'] for v in results.values()]
        always_active_vals = [v['always_active'] for v in results.values()]
        agg = {
            "n_layers": len(results),
            "mean_active_03": round(float(np.mean(active_03_vals)), 4),
            "std_active_03": round(float(np.std(active_03_vals)), 4),
            "mean_eff_dim_90": round(float(np.mean(eff_dim_vals)), 1),
            "std_eff_dim_90": round(float(np.std(eff_dim_vals)), 1),
            "mean_always_active": round(float(np.mean(always_active_vals)), 1),
            "intermediate_dim": intermediate_dim,
            "hidden_dim": hidden_dim,
        }
        print(f"  汇总: mean_active_0.3={agg['mean_active_03']:.3f}, "
              f"mean_eff_dim_90={agg['mean_eff_dim_90']:.1f}, "
              f"mean_always_active={agg['mean_always_active']:.1f}/{intermediate_dim}")
    else:
        agg = {"n_layers": 0}

    return {"stage619": {"layers": results, "aggregate": agg}}


# ============ Stage620: Gate-Up交互分析 ============

def run_stage620(model, tokenizer, model_key):
    """
    分析gate和up之间的逐元素交互关系。

    原理：
    1. 计算每个输入的gate_val和up_val
    2. 逐元素相关分析：gate_i和up_i是否相关？
    3. 门控效率：gate值分布如何影响输出能量
    4. 分析"gate作为信息瓶颈"的程度
    """
    print(f"\n  --- Stage620: Gate-Up交互分析 ---")
    t0 = time.time()
    device = safe_get_device(model)
    layers = discover_layers(model)
    n_layers = len(layers)
    act_name = get_activation_fn(model)

    # Sample layers
    sample_layers = list(range(0, n_layers, max(1, n_layers // 8)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)

    # Collect MLP inputs
    layer_inputs = {}
    for sent in PROBE_SENTENCES:
        enc = tokenizer(sent, return_tensors="pt", truncation=True, max_length=128)
        enc = move_to_device(enc, model)

        local_inputs = {}
        hooks = []
        for li in sample_layers:
            layer = layers[li]
            _, mlp_mod = find_components(layer)
            if mlp_mod is None:
                continue

            def make_pre_hook(layer_idx):
                def pre_hook(module, input):
                    if isinstance(input, tuple) and len(input) > 0:
                        x = input[0]
                        if x.dim() >= 2:
                            local_inputs[layer_idx] = x[0, -1, :].float().detach().cpu()
                return pre_hook

            hooks.append(mlp_mod.register_forward_pre_hook(make_pre_hook(li)))

        try:
            with torch.no_grad():
                model(**enc)
        finally:
            for h in hooks:
                h.remove()

        for li in sample_layers:
            if li in local_inputs:
                layer_inputs.setdefault(li, []).append(local_inputs[li])

    results = {}
    for li in sample_layers:
        if li not in layer_inputs or len(layer_inputs[li]) < 3:
            continue

        layer = layers[li]
        _, mlp_mod = find_components(layer)
        if mlp_mod is None:
            continue

        W_gate, W_up, W_down, is_merged = get_mlp_weights(mlp_mod, act_name)
        if W_gate is None:
            continue

        inputs = torch.stack(layer_inputs[li], dim=0)  # [N, hidden]

        # Compute gate and up
        gate_pre = inputs @ W_gate.T
        gate_act = apply_activation(gate_pre, act_name)

        if W_up is not None:
            up_act = inputs @ W_up.T
        else:
            up_act = gate_pre

        gated = gate_act * up_act

        # 1. Element-wise correlation between gate and up (across neurons)
        # For each input sample, compute correlation between gate_act and up_act
        correlations = []
        for i in range(gate_act.shape[0]):
            g = gate_act[i].numpy()
            u = up_act[i].numpy()
            if np.std(g) > 1e-10 and np.std(u) > 1e-10:
                corr = np.corrcoef(g, u)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

        mean_corr = float(np.mean(correlations)) if correlations else 0
        std_corr = float(np.std(correlations)) if correlations else 0

        # 2. Gate-up direction correlation (vector-level)
        # Are neurons with high gate values also high in up values?
        gate_mean_per_neuron = gate_act.mean(dim=0)  # [intermediate]
        up_mean_per_neuron = up_act.mean(dim=0)
        if torch.std(gate_mean_per_neuron) > 1e-10 and torch.std(up_mean_per_neuron) > 1e-10:
            neuron_corr = float(torch.corrcoef(
                torch.stack([gate_mean_per_neuron, up_mean_per_neuron])
            )[0, 1])
        else:
            neuron_corr = 0

        # 3. Gating efficiency: what fraction of up energy survives gating?
        up_energy = (up_act ** 2).sum(dim=1).mean().item()
        gated_energy = (gated ** 2).sum(dim=1).mean().item()
        gating_efficiency = gated_energy / up_energy if up_energy > 1e-10 else 0

        # 4. Information bottleneck analysis
        # Sort neurons by gate value, compute cumulative output energy
        N = gate_act.shape[0]
        all_gate = gate_act.flatten()
        sorted_vals, sorted_idx = torch.sort(all_gate, descending=True)

        # Compute output energy for top-k% neurons
        energy_by_gate_threshold = {}
        for pct in [1, 5, 10, 20, 50]:
            k = max(1, int(len(sorted_vals) * pct / 100))
            top_k_idx = sorted_idx[:k]
            top_k_energy = (gated.flatten()[top_k_idx] ** 2).sum().item()
            total_energy = (gated ** 2).sum().item()
            energy_by_gate_threshold[f"top_{pct}pct_energy"] = (
                round(top_k_energy / total_energy, 4) if total_energy > 1e-10 else 0
            )

        # 5. Gate activation vs output energy per neuron
        gate_per_neuron_energy = (gate_act ** 2 * up_act ** 2).mean(dim=0)  # [intermediate]
        gate_per_neuron_mean = gate_act.mean(dim=0)
        if torch.std(gate_per_neuron_mean) > 1e-10 and torch.std(gate_per_neuron_energy) > 1e-10:
            gate_energy_corr = float(torch.corrcoef(
                torch.stack([gate_per_neuron_mean, gate_per_neuron_energy])
            )[0, 1])
        else:
            gate_energy_corr = 0

        # 6. "Effective sparsity": what % of neurons account for 90% of output energy?
        neuron_energies = (gated ** 2).mean(dim=0)  # [intermediate]
        sorted_energies, _ = torch.sort(neuron_energies, descending=True)
        cum_energy = torch.cumsum(sorted_energies, dim=0)
        total_e = cum_energy[-1].item()
        if total_e > 1e-10:
            n_for_90pct = (cum_energy < 0.90 * total_e).sum().item() + 1
            n_for_99pct = (cum_energy < 0.99 * total_e).sum().item() + 1
        else:
            n_for_90pct = n_for_99pct = 0

        layer_data = {
            "mean_gate_up_corr": round(mean_corr, 4),
            "std_gate_up_corr": round(std_corr, 4),
            "neuron_level_corr": round(neuron_corr, 4),
            "gating_efficiency": round(gating_efficiency, 4),
            "gate_energy_corr": round(gate_energy_corr, 4),
            "n_neurons_90pct_energy": n_for_90pct,
            "n_neurons_99pct_energy": n_for_99pct,
        }
        for k, v in energy_by_gate_threshold.items():
            layer_data[k] = v

        results[str(li)] = layer_data
        print(f"    L{li}: gate_up_corr={mean_corr:.3f}, "
              f"gating_eff={gating_efficiency:.3f}, "
              f"neurons_90%={n_for_90pct}, "
              f"top5%_energy={energy_by_gate_threshold.get('top_5pct_energy', 0):.3f}")

        del gate_pre, gate_act, up_act, gated
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"  Stage620完成，耗时{elapsed:.1f}s，分析{len(results)}层")

    if results:
        corr_vals = [v['mean_gate_up_corr'] for v in results.values()]
        eff_vals = [v['gating_efficiency'] for v in results.values()]
        n90_vals = [v['n_neurons_90pct_energy'] for v in results.values()]
        agg = {
            "n_layers": len(results),
            "mean_gate_up_corr": round(float(np.mean(corr_vals)), 4),
            "mean_gating_efficiency": round(float(np.mean(eff_vals)), 4),
            "mean_neurons_90pct": round(float(np.mean(n90_vals)), 1),
            "top5pct_energy": round(float(np.mean([v.get('top_5pct_energy', 0) for v in results.values()])), 4),
        }
        print(f"  汇总: mean_gate_up_corr={agg['mean_gate_up_corr']:.3f}, "
              f"mean_gating_eff={agg['mean_gating_efficiency']:.3f}, "
              f"mean_neurons_90%={agg['mean_neurons_90pct']:.1f}")
    else:
        agg = {"n_layers": 0}

    return {"stage620": {"layers": results, "aggregate": agg}}


# ============ Stage621: MLP分段线性近似 ============

def run_stage621(model, tokenizer, model_key):
    """
    分析MLP变换是否存在分段线性结构。

    原理：
    1. 将neurons按gate值分为"活跃"(>0.5)、"半活跃"(0.1-0.5)、"沉默"(<0.1)
    2. 在每组内拟合输入→输出的线性变换
    3. 检查活跃组是否构成低秩子空间
    4. 验证"稀疏激活→低维变换"的因果链
    """
    print(f"\n  --- Stage621: MLP分段线性近似 ---")
    t0 = time.time()
    device = safe_get_device(model)
    layers = discover_layers(model)
    n_layers = len(layers)
    act_name = get_activation_fn(model)

    # Sample layers
    sample_layers = list(range(0, n_layers, max(1, n_layers // 8)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)

    # Collect MLP inputs AND outputs
    layer_inputs = {}
    layer_outputs = {}
    for sent in PROBE_SENTENCES:
        enc = tokenizer(sent, return_tensors="pt", truncation=True, max_length=128)
        enc = move_to_device(enc, model)

        local_inputs = {}
        local_outputs = {}
        hooks = []
        for li in sample_layers:
            layer = layers[li]
            _, mlp_mod = find_components(layer)
            if mlp_mod is None:
                continue

            def make_hooks(layer_idx):
                def pre_hook(module, input):
                    if isinstance(input, tuple) and len(input) > 0:
                        x = input[0]
                        if x.dim() >= 2:
                            local_inputs[layer_idx] = x[0, -1, :].float().detach().cpu()
                def post_hook(module, input, output):
                    if isinstance(output, tuple):
                        local_outputs[layer_idx] = output[0][0, -1, :].float().detach().cpu()
                    else:
                        local_outputs[layer_idx] = output[0, -1, :].float().detach().cpu()
                return pre_hook, post_hook

            pre_h, post_h = make_hooks(li)
            hooks.append(mlp_mod.register_forward_pre_hook(pre_h))
            hooks.append(mlp_mod.register_forward_hook(post_h))

        try:
            with torch.no_grad():
                model(**enc)
        finally:
            for h in hooks:
                h.remove()

        for li in sample_layers:
            if li in local_inputs and li in local_outputs:
                layer_inputs.setdefault(li, []).append(local_inputs[li])
                layer_outputs.setdefault(li, []).append(local_outputs[li])

    results = {}
    for li in sample_layers:
        if li not in layer_inputs or len(layer_inputs[li]) < 3:
            continue

        layer = layers[li]
        _, mlp_mod = find_components(layer)
        if mlp_mod is None:
            continue

        W_gate, W_up, W_down, is_merged = get_mlp_weights(mlp_mod, act_name)
        if W_gate is None:
            continue

        inputs = torch.stack(layer_inputs[li], dim=0)  # [N, hidden]
        outputs = torch.stack(layer_outputs[li], dim=0)  # [N, hidden]
        N = inputs.shape[0]
        hidden = inputs.shape[1]

        # Compute gate activations
        gate_pre = inputs @ W_gate.T
        gate_act = apply_activation(gate_pre, act_name)  # [N, intermediate]

        if W_up is not None:
            up_act = inputs @ W_up.T
        else:
            up_act = gate_pre

        # MLP delta = output - input (residual connection)
        deltas = outputs - inputs  # [N, hidden]

        # === Analysis 1: Piecewise structure by gate threshold ===
        # For each neuron, classify as active/semi-active/silent based on mean gate value
        gate_mean_per_neuron = gate_act.mean(dim=0)  # [intermediate]
        active_mask = gate_mean_per_neuron > 0.5
        semi_active_mask = (gate_mean_per_neuron > 0.1) & (gate_mean_per_neuron <= 0.5)
        silent_mask = gate_mean_per_neuron <= 0.1

        n_active = active_mask.sum().item()
        n_semi = semi_active_mask.sum().item()
        n_silent = silent_mask.sum().item()

        # === Analysis 2: Active neuron subspace dimension ===
        # Project gated output (gate * up) onto the active dimensions
        gated = gate_act * up_act  # [N, intermediate]
        active_gated = gated[:, active_mask] if n_active > 0 else gated

        if active_gated.shape[1] >= N and N >= 3:
            sv_active = torch.linalg.svdvals(active_gated)
            active_energy = (sv_active ** 2).sum().item()
            if active_energy > 1e-10:
                cum = torch.cumsum(sv_active ** 2, dim=0) / active_energy
                active_eff_dim_90 = (cum < 0.90).sum().item() + 1
                active_eff_dim_99 = (cum < 0.99).sum().item() + 1
            else:
                active_eff_dim_90 = active_eff_dim_99 = 0
        else:
            active_eff_dim_90 = active_eff_dim_99 = min(N, active_gated.shape[1]) if n_active > 0 else 0

        # === Analysis 3: Linear fit quality in active subspace ===
        # Fit a linear map from input to delta using only active neurons
        # delta ≈ X @ A, where X = [N, hidden], delta = [N, hidden]
        if N >= 3:
            X = inputs  # [N, hidden]
            Y = deltas  # [N, hidden]

            # Least squares: A = (X^T X)^{-1} X^T Y
            try:
                # Use pseudo-inverse for stability
                XtX = X.T @ X  # [hidden, hidden]
                XtY = X.T @ Y  # [hidden, hidden]
                A = torch.linalg.solve(XtX + 1e-6 * torch.eye(hidden), XtY)

                # Compute fit quality
                Y_pred = X @ A
                residual = Y - Y_pred
                rel_error = (torch.norm(residual) / (torch.norm(Y) + 1e-10)).item()

                # SVD of fitted linear map
                sv_A = torch.linalg.svdvals(A)
                A_energy = (sv_A ** 2).sum().item()
                if A_energy > 1e-10:
                    cum_A = torch.cumsum(sv_A ** 2, dim=0) / A_energy
                    linear_eff_dim_90 = (cum_A < 0.90).sum().item() + 1
                    linear_eff_dim_99 = (cum_A < 0.99).sum().item() + 1
                else:
                    linear_eff_dim_90 = linear_eff_dim_99 = 0

                # Rank of fitted map (singular values > 1% of max)
                if sv_A[0] > 1e-10:
                    fitted_rank = (sv_A > 0.01 * sv_A[0]).sum().item()
                else:
                    fitted_rank = 0
            except Exception as e:
                print(f"    L{li}: Linear fit failed: {e}")
                rel_error = 1.0
                linear_eff_dim_90 = linear_eff_dim_99 = 0
                fitted_rank = 0
        else:
            rel_error = 1.0
            linear_eff_dim_90 = linear_eff_dim_99 = 0
            fitted_rank = 0

        # === Analysis 4: Gate value → output contribution (nonlinearity measure) ===
        # For each neuron, compute how much gate value varies across inputs
        gate_per_neuron_std = gate_act.std(dim=0)  # [intermediate]
        # High variation → gate acts as dynamic selector
        # Low variation → gate is approximately constant (linear regime)
        mean_gate_variation = gate_per_neuron_std.mean().item()
        high_variation_neurons = (gate_per_neuron_std > 0.2).sum().item()

        # === Analysis 5: Compare linear approximation with/without gating ===
        # Without gating: delta_no_gate ≈ (up_act) @ W_down
        # With gating: delta_with_gate = (gate_act * up_act) @ W_down
        if W_down is not None:
            # Reconstruct output without gating
            up_projected = up_act @ W_down.T  # [N, hidden] (approximate, ignoring layernorm)
            gated_projected = gated @ W_down.T  # [N, hidden]

            # How different are they?
            no_gate_error = (torch.norm(gated_projected - up_projected) / (torch.norm(gated_projected) + 1e-10)).item()
            # The ratio tells us how much gating matters
            gating_importance = 1.0 - min(no_gate_error, 1.0)
        else:
            gating_importance = 0
            no_gate_error = 0

        layer_data = {
            "n_active": n_active,
            "n_semi_active": n_semi,
            "n_silent": n_silent,
            "active_ratio": round(n_active / (n_active + n_semi + n_silent + 1e-10), 4),
            "silent_ratio": round(n_silent / (n_active + n_semi + n_silent + 1e-10), 4),
            "active_eff_dim_90": active_eff_dim_90,
            "active_eff_dim_99": active_eff_dim_99,
            "linear_fit_rel_error": round(rel_error, 4),
            "linear_eff_dim_90": linear_eff_dim_90,
            "linear_eff_dim_99": linear_eff_dim_99,
            "fitted_rank": fitted_rank,
            "mean_gate_variation": round(mean_gate_variation, 4),
            "high_variation_neurons": high_variation_neurons,
            "gating_importance": round(gating_importance, 4),
            "no_gate_error": round(no_gate_error, 4),
        }

        results[str(li)] = layer_data
        print(f"    L{li}: active={n_active}, silent={n_silent}, "
              f"active_eff_dim_90={active_eff_dim_90}, "
              f"linear_fit_err={rel_error:.3f}, "
              f"fitted_rank={fitted_rank}, "
              f"gating_importance={gating_importance:.3f}")

        del gate_pre, gate_act, up_act, gated, deltas
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    print(f"  Stage621完成，耗时{elapsed:.1f}s，分析{len(results)}层")

    if results:
        active_vals = [v['active_ratio'] for v in results.values()]
        silent_vals = [v['silent_ratio'] for v in results.values()]
        err_vals = [v['linear_fit_rel_error'] for v in results.values()]
        rank_vals = [v['fitted_rank'] for v in results.values()]
        gate_vals = [v['gating_importance'] for v in results.values()]
        agg = {
            "n_layers": len(results),
            "mean_active_ratio": round(float(np.mean(active_vals)), 4),
            "mean_silent_ratio": round(float(np.mean(silent_vals)), 4),
            "mean_linear_fit_error": round(float(np.mean(err_vals)), 4),
            "mean_fitted_rank": round(float(np.mean(rank_vals)), 1),
            "mean_gating_importance": round(float(np.mean(gate_vals)), 4),
        }
        print(f"  汇总: mean_active={agg['mean_active_ratio']:.3f}, "
              f"mean_silent={agg['mean_silent_ratio']:.3f}, "
              f"mean_linear_err={agg['mean_linear_fit_error']:.3f}, "
              f"mean_rank={agg['mean_fitted_rank']:.1f}, "
              f"mean_gating_imp={agg['mean_gating_importance']:.3f}")
    else:
        agg = {"n_layers": 0}

    return {"stage621": {"layers": results, "aggregate": agg}}


# ============ Main ============

def main():
    model_key = sys.argv[1].lower().strip() if len(sys.argv) > 1 else "qwen3"
    print(f"{'='*60}")
    print(f"Stage619-621: SwiGLU维度压缩 + Gate-Up交互 + 分段线性近似")
    print(f"模型: {model_key}")
    print(f"{'='*60}")

    t_total = time.time()

    # Load model
    print(f"\n加载模型 {model_key}...")
    model, tokenizer = load_model_bundle(model_key)
    device = safe_get_device(model)
    print(f"  设备: {device}")

    # Run all stages
    all_results = {}

    r619 = run_stage619(model, tokenizer, model_key)
    all_results.update(r619)

    gc.collect()
    torch.cuda.empty_cache()

    r620 = run_stage620(model, tokenizer, model_key)
    all_results.update(r620)

    gc.collect()
    torch.cuda.empty_cache()

    r621 = run_stage621(model, tokenizer, model_key)
    all_results.update(r621)

    # Save results
    output_path = OUTPUT_DIR / f"stage619_620_621_{model_key}_{TIMESTAMP}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存到: {output_path}")

    # Final summary
    elapsed_total = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"全部完成！总耗时: {elapsed_total:.1f}s")
    print(f"{'='*60}")

    # Print key findings
    print("\n=== 关键发现摘要 ===")
    if "stage619" in all_results:
        a = all_results["stage619"]["aggregate"]
        print(f"\n[Stage619] SwiGLU稀疏激活:")
        print(f"  mean_active(gate>0.3) = {a.get('mean_active_03', 'N/A')}")
        print(f"  mean_eff_dim_90 = {a.get('mean_eff_dim_90', 'N/A')}")
        print(f"  mean_always_active = {a.get('mean_always_active', 'N/A')}/{a.get('intermediate_dim', 'N/A')}")

    if "stage620" in all_results:
        a = all_results["stage620"]["aggregate"]
        print(f"\n[Stage620] Gate-Up交互:")
        print(f"  mean_gate_up_corr = {a.get('mean_gate_up_corr', 'N/A')}")
        print(f"  mean_gating_efficiency = {a.get('mean_gating_efficiency', 'N/A')}")
        print(f"  mean_neurons_90%_energy = {a.get('mean_neurons_90pct', 'N/A')}")
        print(f"  top5%_energy = {a.get('top5pct_energy', 'N/A')}")

    if "stage621" in all_results:
        a = all_results["stage621"]["aggregate"]
        print(f"\n[Stage621] MLP分段线性:")
        print(f"  mean_active_ratio = {a.get('mean_active_ratio', 'N/A')}")
        print(f"  mean_silent_ratio = {a.get('mean_silent_ratio', 'N/A')}")
        print(f"  mean_linear_fit_error = {a.get('mean_linear_fit_error', 'N/A')}")
        print(f"  mean_fitted_rank = {a.get('mean_fitted_rank', 'N/A')}")
        print(f"  mean_gating_importance = {a.get('mean_gating_importance', 'N/A')}")

    # Free model
    free_model(model)
    print("\n模型已释放。")


if __name__ == "__main__":
    main()
