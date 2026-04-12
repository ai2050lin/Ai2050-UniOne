"""
Phase LXXV-P384/385/386: 中间层正交性验证 + 叠加效应机制 + lambda物理含义
======================================================================

核心目标: 攻克Phase LXXIV留下的三个关键问题

P384: 中间层三维度正交性验证 ★★★最关键实验★★★
  - 在L0, L5, L10, L15, L20的隐藏层空间，计算style/logic/grammar方向的正交性
  - 方法: 将W_lm空间的三个维度方向注入L0，然后在每层捕获hidden state
  - 计算hidden state中三个维度的方向变化(Δh_style, Δh_logic, Δh_grammar)
  - 如果中间层的cos(Δh_style, Δh_logic)仍然<0.1 → 正交性贯穿全网络
  - 如果中间层的cos>0.3 → 正交性在某层被破坏

P385: 叠加效应3.8x的机制解析
  - P380发现: 联合注入style+logic→叠加比3.8x，远超线性叠加1.0
  - 假设1(正交增强): 两个正交方向的联合注入减少了每个方向在非目标维度的泄漏
  - 假设2(非线性交互): MLP的SiLU激活函数导致交叉项放大
  - 假设3(范数效应): 联合注入的范数更大，触发了更强的logit响应
  - 实验: 精确控制注入范数，分离线性项和非线性交互项

P386: lambda物理含义解析
  - P383发现: λ≈0.023跨模型一致
  - 假设1: λ = c/d_model (与模型维度成反比)
  - 假设2: λ = c/sigma_lm (与lm_head的范数有关)
  - 假设3: λ与SiLU工作点有关
  - 实验: 计算d_model, sigma_lm, SiLU统计量，看哪个与λ相关

实验模型: qwen3 -> glm4 -> deepseek7b (串行)
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


# ========== P384: 中间层三维度正交性验证 ==========

def run_p384(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P384: Mid-layer orthogonality verification - {model_name}")
    print(f"{'='*60}")

    results = {}
    embed = model.get_input_embeddings()

    dim_names = ["style", "logic", "grammar"]
    dim_pairs = [STYLE_PAIRS[0], LOGIC_PAIRS[0], GRAMMAR_PAIRS[0]]
    beta = 8.0

    n_layers_total = len(get_layers(model))
    # 选择扫描的层
    scan_layers = [0, 1, 2, 5, 10, 15, 20]
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

    # 对每个维度分别注入，捕获每层hidden state
    for dim_name, (pos, neg) in zip(dim_names, dim_pairs):
        direction = dim_directions[dim_name]["direction"]
        w_tensor = torch.tensor(direction, dtype=torch.float32, device=device)

        inputs_embeds_int = inputs_embeds_base.clone()
        inputs_embeds_int[0, -1, :] += (beta * w_tensor).to(model.dtype)

        # 捕获每层输出
        captured = {}

        def make_hook(key):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    captured[key] = output[0].detach().float()
                else:
                    captured[key] = output.detach().float()
            return hook

        handles = []
        for i, layer in enumerate(layers):
            if i in scan_layers:
                handles.append(layer.register_forward_hook(make_hook(f"L{i}")))

        with torch.no_grad():
            _ = model(inputs_embeds=inputs_embeds_int, position_ids=position_ids)

        for h in handles:
            h.remove()

        # 保存每层的hidden state方向变化
        dim_directions[dim_name]["layer_states"] = {}
        for l in scan_layers:
            key = f"L{l}"
            if key in captured:
                h_state = captured[key][0, -1, :].cpu().numpy()
                dim_directions[dim_name]["layer_states"][l] = h_state

    # 同样捕获baseline
    captured_base = {}

    def make_hook_base(key):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured_base[key] = output[0].detach().float()
            else:
                captured_base[key] = output.detach().float()
        return hook

    handles_base = []
    for i, layer in enumerate(layers):
        if i in scan_layers:
            handles_base.append(layer.register_forward_hook(make_hook_base(f"L{i}")))

    with torch.no_grad():
        _ = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids)

    for h in handles_base:
        h.remove()

    # 计算每层的Δh方向
    delta_h = {}
    for dim_name in dim_names:
        delta_h[dim_name] = {}
        for l in scan_layers:
            key = f"L{l}"
            if l in dim_directions[dim_name]["layer_states"] and key in captured_base:
                h_int = dim_directions[dim_name]["layer_states"][l]
                h_base = captured_base[key][0, -1, :].cpu().numpy()
                dh = h_int - h_base
                dh_norm = np.linalg.norm(dh)
                if dh_norm > 1e-8:
                    delta_h[dim_name][l] = dh / dh_norm  # 归一化方向
                else:
                    delta_h[dim_name][l] = np.zeros_like(dh)

    # 计算每层维度间的正交性
    layer_orthogonality = {}
    for l in scan_layers:
        ortho = {}
        for i, n1 in enumerate(dim_names):
            for j, n2 in enumerate(dim_names):
                if i >= j:
                    continue
                if l in delta_h[n1] and l in delta_h[n2]:
                    cos_val = float(np.dot(delta_h[n1][l], delta_h[n2][l]))
                    ortho[f"cos({n1},{n2})"] = cos_val
        layer_orthogonality[f"L{l}"] = ortho
        ortho_str = ", ".join(f"{k}={v:.4f}" for k, v in ortho.items())
        print(f"  L{l}: {ortho_str}")

    # 同时计算每个维度方向在W_lm空间中的投影(cos与原始方向)
    dim_preservation = {}
    for dim_name in dim_names:
        preservation = {}
        w_lm_dir = dim_directions[dim_name]["direction"]
        for l in scan_layers:
            if l in delta_h[dim_name]:
                cos_wlm = float(np.dot(delta_h[dim_name][l], w_lm_dir))
                preservation[f"L{l}"] = cos_wlm
        dim_preservation[dim_name] = preservation
        pres_str = ", ".join(f"L{l}={v:.4f}" for l, v in sorted(preservation.items()))
        print(f"  {dim_name} preservation: {pres_str}")

    results = {
        "layer_orthogonality": layer_orthogonality,
        "dim_preservation": dim_preservation,
        "scan_layers": scan_layers,
    }

    return results


# ========== P385: 叠加效应3.8x的机制解析 ==========

def run_p385(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P385: Superposition effect mechanism - {model_name}")
    print(f"{'='*60}")

    results = {}
    embed = model.get_input_embeddings()

    dim_names = ["style", "logic", "grammar"]
    dim_pairs = [STYLE_PAIRS[0], LOGIC_PAIRS[0], GRAMMAR_PAIRS[0]]

    # 计算维度方向
    dim_directions = {}
    for name, (pos, neg) in zip(dim_names, dim_pairs):
        direction, norm = get_dimension_direction(model, tokenizer, pos, neg)
        dim_directions[name] = {"direction": direction, "norm": norm, "pos": pos, "neg": neg}

    # 基线前向
    toks = tokenizer(PROMPT, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    seq_len = input_ids.shape[1]
    inputs_embeds_base = embed(input_ids).detach().clone().to(model.dtype)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    with torch.no_grad():
        logits_base = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids).logits[0, -1, :]

    # 实验1: 范数控制实验
    # 如果叠加比是因为范数效应，那么控制范数后叠加比应接近1.0
    print(f"\n  --- Exp1: Norm-controlled superposition ---")
    beta_single = 5.0
    beta_combined = 5.0 / np.sqrt(3)  # 控制联合注入的范数=单独注入

    single_effects = {}
    for name in dim_names:
        direction = dim_directions[name]["direction"]
        w_tensor = torch.tensor(direction, dtype=torch.float32, device=device)
        inputs_embeds_int = inputs_embeds_base.clone()
        inputs_embeds_int[0, -1, :] += (beta_single * w_tensor).to(model.dtype)

        with torch.no_grad():
            logits_int = model(inputs_embeds=inputs_embeds_int, position_ids=position_ids).logits[0, -1, :]

        pos_id = tokenizer.encode(dim_directions[name]["pos"], add_special_tokens=False)[0]
        neg_id = tokenizer.encode(dim_directions[name]["neg"], add_special_tokens=False)[0]
        delta_pos = float(logits_int[pos_id].cpu() - logits_base[pos_id].cpu())
        delta_neg = float(logits_int[neg_id].cpu() - logits_base[neg_id].cpu())
        single_effects[name] = {"delta_pos": delta_pos, "delta_neg": delta_neg, "delta_diff": delta_pos - delta_neg}
        print(f"  {name} single(beta={beta_single}): diff={delta_pos - delta_neg:.3f}")

    # 联合注入(范数控制)
    combined_dir = sum(dim_directions[n]["direction"] for n in dim_names)
    combined_norm = np.linalg.norm(combined_dir)
    combined_dir_normed = combined_dir / combined_norm

    w_tensor = torch.tensor(combined_dir_normed, dtype=torch.float32, device=device)
    inputs_embeds_int = inputs_embeds_base.clone()
    inputs_embeds_int[0, -1, :] += (beta_single * w_tensor).to(model.dtype)  # 相同范数

    with torch.no_grad():
        logits_int = model(inputs_embeds=inputs_embeds_int, position_ids=position_ids).logits[0, -1, :]

    norm_controlled = {}
    for name in dim_names:
        pos_id = tokenizer.encode(dim_directions[name]["pos"], add_special_tokens=False)[0]
        neg_id = tokenizer.encode(dim_directions[name]["neg"], add_special_tokens=False)[0]
        delta_pos = float(logits_int[pos_id].cpu() - logits_base[pos_id].cpu())
        delta_neg = float(logits_int[neg_id].cpu() - logits_base[neg_id].cpu())
        norm_controlled[name] = {"delta_pos": delta_pos, "delta_neg": delta_neg, "delta_diff": delta_pos - delta_neg}
        print(f"  {name} norm-controlled: diff={delta_pos - delta_neg:.3f}")

    # 实验2: 正交独立注入(每个维度分别注入beta)
    print(f"\n  --- Exp2: Orthogonal independent injection ---")
    ortho_dir = sum(dim_directions[n]["direction"] * beta_single for n in dim_names)
    w_tensor = torch.tensor(ortho_dir, dtype=torch.float32, device=device)
    inputs_embeds_int = inputs_embeds_base.clone()
    inputs_embeds_int[0, -1, :] += w_tensor.to(model.dtype)

    with torch.no_grad():
        logits_int = model(inputs_embeds=inputs_embeds_int, position_ids=position_ids).logits[0, -1, :]

    ortho_independent = {}
    for name in dim_names:
        pos_id = tokenizer.encode(dim_directions[name]["pos"], add_special_tokens=False)[0]
        neg_id = tokenizer.encode(dim_directions[name]["neg"], add_special_tokens=False)[0]
        delta_pos = float(logits_int[pos_id].cpu() - logits_base[pos_id].cpu())
        delta_neg = float(logits_int[neg_id].cpu() - logits_base[neg_id].cpu())
        superposition_ratio = (delta_pos - delta_neg) / single_effects[name]["delta_diff"] if abs(single_effects[name]["delta_diff"]) > 0.01 else 0.0
        ortho_independent[name] = {
            "delta_pos": delta_pos, "delta_neg": delta_neg,
            "delta_diff": delta_pos - delta_neg,
            "superposition_ratio": superposition_ratio,
        }
        print(f"  {name} ortho-indep: diff={delta_pos - delta_neg:.3f}, ratio={superposition_ratio:.3f}")

    # 实验3: 逐步增加维度，看logit变化的线性/非线性
    print(f"\n  --- Exp3: Incremental dimension addition ---")
    import itertools
    incremental = {}
    betas = [1.0, 2.0, 5.0, 10.0]

    for beta in betas:
        beta_results = {}
        # 单维度
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
            beta_results[f"{name}_single"] = delta_pos - delta_neg

        # 三维度联合
        ortho_dir = sum(dim_directions[n]["direction"] * beta for n in dim_names)
        w_tensor = torch.tensor(ortho_dir, dtype=torch.float32, device=device)
        inputs_embeds_int = inputs_embeds_base.clone()
        inputs_embeds_int[0, -1, :] += w_tensor.to(model.dtype)

        with torch.no_grad():
            logits_int = model(inputs_embeds=inputs_embeds_int, position_ids=position_ids).logits[0, -1, :]

        for name in dim_names:
            pos_id = tokenizer.encode(dim_directions[name]["pos"], add_special_tokens=False)[0]
            neg_id = tokenizer.encode(dim_directions[name]["neg"], add_special_tokens=False)[0]
            delta_pos = float(logits_int[pos_id].cpu() - logits_base[pos_id].cpu())
            delta_neg = float(logits_int[neg_id].cpu() - logits_base[neg_id].cpu())
            beta_results[f"{name}_combined"] = delta_pos - delta_neg

        # 计算交互项 = 联合效果 - 线性叠加
        for name in dim_names:
            interaction = beta_results[f"{name}_combined"] - beta_results[f"{name}_single"]
            beta_results[f"{name}_interaction"] = interaction

        incremental[f"beta_{beta}"] = beta_results
        print(f"  beta={beta}: " + ", ".join(
            f"{name}: single={beta_results[f'{name}_single']:.2f}, "
            f"combined={beta_results[f'{name}_combined']:.2f}, "
            f"interaction={beta_results[f'{name}_interaction']:.2f}"
            for name in dim_names
        ))

    results = {
        "single_effects": single_effects,
        "norm_controlled": norm_controlled,
        "ortho_independent": ortho_independent,
        "incremental": incremental,
    }

    return results


# ========== P386: lambda物理含义解析 ==========

def run_p386(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P386: Lambda physical meaning analysis - {model_name}")
    print(f"{'='*60}")

    results = {}
    embed = model.get_input_embeddings()
    attr = "red"

    # 1. 模型基础参数
    n_layers_total = len(get_layers(model))
    hidden_dim = embed.weight.shape[1]  # d_model
    vocab_size = embed.weight.shape[0]

    print(f"  d_model(hidden_dim) = {hidden_dim}")
    print(f"  n_layers = {n_layers_total}")
    print(f"  vocab_size = {vocab_size}")

    # 2. lm_head的统计特性
    lm_head = model.lm_head
    W_lm = lm_head.weight.detach().cpu().float()  # [vocab, hidden]
    W_lm_norms = W_lm.norm(dim=1)
    W_lm_mean_norm = float(W_lm_norms.mean())
    W_lm_std_norm = float(W_lm_norms.std())

    # W_lm的谱范数
    try:
        _, S_lm, _ = torch.linalg.svd(W_lm.float(), full_matrices=False)
        sigma_max_lm = float(S_lm[0].item())
        sigma_min_lm = float(S_lm[-1].item())
        condition_number = sigma_max_lm / max(sigma_min_lm, 1e-10)
    except:
        sigma_max_lm = float(W_lm.norm().item())
        sigma_min_lm = 0.0
        condition_number = 0.0

    print(f"  W_lm mean_norm = {W_lm_mean_norm:.4f}")
    print(f"  W_lm std_norm = {W_lm_std_norm:.4f}")
    print(f"  W_lm sigma_max = {sigma_max_lm:.4f}")
    print(f"  W_lm condition_number = {condition_number:.4f}")

    # 3. W_up的统计特性(每层)
    layers = get_layers(model)
    w_up_stats = []
    for i, layer in enumerate(layers):
        mlp = get_mlp(model, layer)
        if hasattr(mlp, "up_proj"):
            W = mlp.up_proj.weight.detach().cpu().float()
        elif hasattr(mlp, "gate_up_proj"):
            W_full = mlp.gate_up_proj.weight.detach().cpu().float()
            inter_size = W_full.shape[0] // 2
            W = W_full[inter_size:, :]
        else:
            w_up_stats.append({"sigma_max": 0, "frobenius": 0})
            continue

        try:
            _, S, _ = torch.linalg.svd(W, full_matrices=False)
            sigma_max = float(S[0].item())
        except:
            sigma_max = float(W.norm().item())

        frob_norm = float(W.norm().item())
        w_up_stats.append({"sigma_max": sigma_max, "frobenius": frob_norm})

    # 4. SiLU工作点统计
    w_lm_norm_np, _ = get_w_lm_normed(model, tokenizer, attr)
    w_lm_tensor = torch.tensor(w_lm_norm_np, dtype=torch.float32, device=device)

    toks = tokenizer(PROMPT, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    seq_len = input_ids.shape[1]
    inputs_embeds = embed(input_ids).detach().clone().to(model.dtype)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    # 捕获每层MLP输入和gate输出
    silu_stats = []
    for i in range(min(n_layers_total, 10)):
        layer = layers[i]
        mlp = get_mlp(model, layer)

        # Hook MLP输入
        captured_mlp_in = {}

        def make_hook_mlp_in(key):
            def hook(module, input, output):
                if isinstance(input, tuple):
                    captured_mlp_in[key] = input[0].detach().float()
            return hook

        if hasattr(layer, "post_attention_layernorm"):
            ln = layer.post_attention_layernorm
        else:
            ln = layer.input_layernorm

        h = ln.register_forward_hook(make_hook_mlp_in("mlp_in"))
        with torch.no_grad():
            _ = model(inputs_embeds=inputs_embeds, position_ids=position_ids)
        h.remove()

        if "mlp_in" not in captured_mlp_in:
            silu_stats.append({"gate_mean": 0, "gate_std": 0, "silu_grad_mean": 0})
            continue

        mlp_input = captured_mlp_in["mlp_in"][0, -1, :].to(mlp.gate_proj.weight.dtype if hasattr(mlp, 'gate_proj') else model.dtype)

        with torch.no_grad():
            if hasattr(mlp, "gate_proj"):
                gate_out = mlp.gate_proj(mlp_input.unsqueeze(0).unsqueeze(0))[0, 0, :].detach().float().cpu().numpy()
            elif hasattr(mlp, "gate_up_proj"):
                W_gate = mlp.gate_up_proj.weight.detach().cpu().float()
                inter_size = W_gate.shape[0] // 2
                W_g = W_gate[:inter_size, :]
                x_cpu = mlp_input.cpu().float()
                gate_out = (W_g @ x_cpu).numpy()
            else:
                silu_stats.append({"gate_mean": 0, "gate_std": 0, "silu_grad_mean": 0})
                continue

        # SiLU导数
        sigmoid_gate = 1.0 / (1.0 + np.exp(-gate_out))
        silu_grad = sigmoid_gate * (1.0 + gate_out * (1.0 - sigmoid_gate))

        silu_stats.append({
            "gate_mean": float(np.mean(gate_out)),
            "gate_std": float(np.std(gate_out)),
            "silu_grad_mean": float(np.mean(silu_grad)),
            "silu_grad_std": float(np.std(silu_grad)),
            "gate_abs_mean": float(np.mean(np.abs(gate_out))),
        })

    # 5. 计算cos衰减曲线和lambda
    n_scan = min(n_layers_total, 16)
    layers_scan = get_layers(model, n_scan)

    captured_base = {}
    captured_int = {}

    def make_hook(storage, key):
        def hook(module, input, output):
            if isinstance(output, tuple):
                storage[key] = output[0].detach().float()
            else:
                storage[key] = output.detach().float()
        return hook

    inputs_embeds_int = inputs_embeds.clone()
    inputs_embeds_int[0, -1, :] += (8.0 * w_lm_tensor).to(model.dtype)

    handles_b = []
    handles_i = []
    for i, layer in enumerate(layers_scan):
        handles_b.append(layer.register_forward_hook(make_hook(captured_base, f"L{i}")))
        handles_i.append(layer.register_forward_hook(make_hook(captured_int, f"L{i}")))

    with torch.no_grad():
        _ = model(inputs_embeds=inputs_embeds, position_ids=position_ids)
    for h in handles_b:
        h.remove()

    with torch.no_grad():
        _ = model(inputs_embeds=inputs_embeds_int, position_ids=position_ids)
    for h in handles_i:
        h.remove()

    # 计算cos曲线
    cos_curve = []
    for i in range(n_scan):
        key = f"L{i}"
        if key not in captured_base or key not in captured_int:
            continue
        h_b = captured_base[key][0, -1, :].cpu().numpy()
        h_i = captured_int[key][0, -1, :].cpu().numpy()
        delta = h_i - h_b
        dn = np.linalg.norm(delta)
        cos_val = float(np.dot(delta, w_lm_norm_np) / dn) if dn > 1e-8 else 0.0
        cos_curve.append({"layer": i, "cos": cos_val})

    # 拟合lambda
    cum_sigma_up = np.cumsum([s["sigma_max"] for s in w_up_stats[:n_scan]])
    if len(cos_curve) >= 3:
        cos0 = cos_curve[0]["cos"] if cos_curve[0]["cos"] > 0.01 else 0.99
        lam_values = []
        for i in range(1, min(len(cos_curve), len(cum_sigma_up))):
            if cos_curve[i]["cos"] > 0.01 and cum_sigma_up[i] > 0:
                lam = -np.log(cos_curve[i]["cos"] / cos0) / cum_sigma_up[i]
                lam_values.append(lam)
        avg_lambda = np.mean(lam_values) if lam_values else 0
    else:
        avg_lambda = 0

    # 6. 分析lambda与各参数的关系
    print(f"\n  === Lambda analysis ===")
    print(f"  lambda = {avg_lambda:.6f}")
    print(f"  d_model = {hidden_dim}")
    print(f"  1/d_model = {1.0/hidden_dim:.6f}")
    print(f"  lambda * d_model = {avg_lambda * hidden_dim:.4f}")
    print(f"  W_lm mean_norm = {W_lm_mean_norm:.4f}")
    print(f"  lambda * W_lm_norm = {avg_lambda * W_lm_mean_norm:.6f}")
    print(f"  W_lm sigma_max = {sigma_max_lm:.4f}")
    print(f"  lambda * sigma_max_lm = {avg_lambda * sigma_max_lm:.6f}")
    if silu_stats:
        avg_silu_grad = np.mean([s["silu_grad_mean"] for s in silu_stats if s["silu_grad_mean"] > 0])
        print(f"  avg SiLU grad = {avg_silu_grad:.4f}")
        print(f"  lambda / SiLU_grad = {avg_lambda / avg_silu_grad:.6f}" if avg_silu_grad > 0 else "  N/A")

    # 7. 中间层W_up的sigma_max与cos衰减的关系
    print(f"\n  === Per-layer W_up sigma_max ===")
    for i in range(min(10, len(w_up_stats))):
        s = w_up_stats[i]
        c = cos_curve[i]["cos"] if i < len(cos_curve) else 0
        print(f"  L{i}: sigma_max={s['sigma_max']:.4f}, cos={c:.4f}, cum_sigma={cum_sigma_up[i]:.4f}")

    results = {
        "d_model": hidden_dim,
        "n_layers": n_layers_total,
        "vocab_size": vocab_size,
        "W_lm_mean_norm": W_lm_mean_norm,
        "W_lm_sigma_max": sigma_max_lm,
        "W_lm_condition_number": condition_number,
        "w_up_stats": w_up_stats,
        "silu_stats": silu_stats,
        "cos_curve": cos_curve,
        "avg_lambda": float(avg_lambda),
        "lambda_times_d_model": float(avg_lambda * hidden_dim),
        "lambda_times_W_lm_norm": float(avg_lambda * W_lm_mean_norm),
        "lambda_times_sigma_max_lm": float(avg_lambda * sigma_max_lm),
    }

    if silu_stats:
        avg_silu_grad = np.mean([s["silu_grad_mean"] for s in silu_stats if s["silu_grad_mean"] > 0])
        results["avg_silu_grad"] = float(avg_silu_grad)
        if avg_silu_grad > 0:
            results["lambda_over_silu_grad"] = float(avg_lambda / avg_silu_grad)

    return results


# ========== Main ==========

def main():
    parser = argparse.ArgumentParser(description="Phase LXXV: Mid-layer Orthogonality + Superposition + Lambda")
    parser.add_argument("--model", type=str, default="qwen3", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=str, default="all", choices=["p384", "p385", "p386", "all"])
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
            if args.exp in ["p384", "all"]:
                r384 = run_p384(model, tokenizer, device, model_name)
                all_results["p384"] = r384

            if args.exp in ["p385", "all"]:
                r385 = run_p385(model, tokenizer, device, model_name)
                all_results["p385"] = r385

            if args.exp in ["p386", "all"]:
                r386 = run_p386(model, tokenizer, device, model_name)
                all_results["p386"] = r386

        except Exception as e:
            print(f"  ERROR in {model_name}: {e}")
            traceback.print_exc()
            all_results["error"] = str(e)

        # Save
        out_file = OUT_DIR / f"phase_lxxv_p384_386_{model_name}_{timestamp}.json"
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
