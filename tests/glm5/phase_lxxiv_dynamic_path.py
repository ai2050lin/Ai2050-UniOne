"""
Phase LXXIV-P380/381/382/383: 动态路径选择假说核心验证
======================================================================

核心目标: 验证"风格×逻辑×语法"三维度路径选择的独立性，以及注意力头的路径选择功能

P380: 三维度正交干预实验 ★★★最关键实验★★★
  - 同时在L0注入"风格方向"(如formal/informal) + "逻辑方向"(如true/false) + "语法方向"(如active/passive)
  - 如果三维度正交(cos<0.1)，则支持独立性假说
  - 如果三维度耦合(cos>0.3)，则推翻独立性假说
  - 单维度干预 vs 双维度干预 vs 三维度干预的效果对比

P381: 注意力头的路径选择功能分析
  - 对L0-L5的每个注意力头，计算其对W_lm[attr]方向的敏感度
  - 看是否存在"风格头"、"逻辑头"、"语法头"的分工
  - 计算每个头对最终cos值的贡献比例

P382: 残差流的路径记忆量化
  - 在每层计算: 残差分量中W_lm[attr]方向的投影 vs MLP分量的投影 vs Attention分量的投影
  - 量化"路径记忆"(残差) vs "路径更新"(MLP+Attention)的比例
  - 验证W_up越大的模型，路径记忆衰减越快

P383: 动态路径选择的数学模型拟合
  - 假设cos(t) = cos(0) * exp(-λ*t)，其中λ = f(W_up, W_down, SiLU工作点)
  - 用三个模型的数据拟合λ的函数形式
  - 验证模型是否能预测方向保持的半衰期

实验模型: qwen3 → glm4 → deepseek7b (串行)
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

# 风格维度词对 (formal/informal)
STYLE_PAIRS = [
    ("formal", "informal"),
    ("polite", "rude"),
    ("elegant", "casual"),
]

# 逻辑/事实维度词对 (true/false)
LOGIC_PAIRS = [
    ("true", "false"),
    ("correct", "wrong"),
    ("real", "fake"),
]

# 语法维度词对 (active/passive, singular/plural)
GRAMMAR_PAIRS = [
    ("active", "passive"),
    ("singular", "plural"),
    ("present", "past"),
]

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
    """获取词的lm_head方向(归一化)"""
    tok_ids = tokenizer.encode(word, add_special_tokens=False)
    tok_id = tok_ids[0]
    lm_head = model.lm_head
    w = lm_head.weight[tok_id].detach().cpu().float()
    w_norm = w / w.norm()
    return w_norm.numpy(), w.numpy()


def get_dimension_direction(model, tokenizer, word_pos, word_neg):
    """获取一个维度的方向向量 = w_pos - w_neg, 归一化"""
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


# ========== P380: 三维度正交干预实验 ==========

def run_p380(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P380: 三维度正交干预实验 - {model_name}")
    print(f"{'='*60}")

    results = {}
    embed = model.get_input_embeddings()

    # 1. 计算三个维度的方向
    dim_names = ["style", "logic", "grammar"]
    dim_pairs = [STYLE_PAIRS[0], LOGIC_PAIRS[0], GRAMMAR_PAIRS[0]]

    dim_directions = {}
    for name, (pos, neg) in zip(dim_names, dim_pairs):
        direction, norm = get_dimension_direction(model, tokenizer, pos, neg)
        dim_directions[name] = {
            "direction": direction,
            "norm": norm,
            "pos_word": pos,
            "neg_word": neg,
        }
        print(f"  {name}维度: {pos}/{neg}, ||方向||={norm:.4f}")

    # 2. 计算维度间的正交性
    print(f"\n  维度间正交性:")
    orthogonality = {}
    for i, n1 in enumerate(dim_names):
        for j, n2 in enumerate(dim_names):
            if i >= j:
                continue
            cos_ij = float(np.dot(dim_directions[n1]["direction"], dim_directions[n2]["direction"]))
            orthogonality[f"{n1}_{n2}"] = cos_ij
            label = "[OK] orthogonal" if abs(cos_ij) < 0.15 else "[X] coupled"
            print(f"  cos({n1}, {n2}) = {cos_ij:.4f} {label}")

    # 3. 扩展到所有词对
    print(f"\n  扩展正交性分析(所有词对):")
    all_pairs_extended = {
        "style_formal": ("formal", "informal"),
        "style_polite": ("polite", "rude"),
        "logic_true": ("true", "false"),
        "logic_correct": ("correct", "wrong"),
        "grammar_active": ("active", "passive"),
        "grammar_singular": ("singular", "plural"),
    }

    all_directions = {}
    for key, (pos, neg) in all_pairs_extended.items():
        direction, norm = get_dimension_direction(model, tokenizer, pos, neg)
        all_directions[key] = {"direction": direction, "norm": norm}
        print(f"  {key}: ||方向||={norm:.4f}")

    # 计算所有维度间的cos
    ext_orthogonality = {}
    keys = list(all_directions.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            cos_ij = float(np.dot(all_directions[keys[i]]["direction"], all_directions[keys[j]]["direction"]))
            ext_orthogonality[f"{keys[i]}_{keys[j]}"] = cos_ij

    # 4. 单维度干预效果
    print(f"\n  单维度干预效果:")
    single_interventions = {}
    beta = 5.0

    for name in dim_names:
        direction = dim_directions[name]["direction"]
        w_tensor = torch.tensor(direction, dtype=torch.float32, device=device)

        toks = tokenizer(PROMPT, return_tensors="pt").to(device)
        input_ids = toks.input_ids
        seq_len = input_ids.shape[1]
        inputs_embeds_base = embed(input_ids).detach().clone().to(model.dtype)

        # 基线
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        with torch.no_grad():
            logits_base = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids).logits[0, -1, :]

        # 干预
        inputs_embeds_int = inputs_embeds_base.clone()
        inputs_embeds_int[0, -1, :] += (beta * w_tensor).to(model.dtype)

        with torch.no_grad():
            logits_int = model(inputs_embeds=inputs_embeds_int, position_ids=position_ids).logits[0, -1, :]

        # 测量: pos词和neg词的logit差
        pos_word = dim_directions[name]["pos_word"]
        neg_word = dim_directions[name]["neg_word"]
        pos_id = tokenizer.encode(pos_word, add_special_tokens=False)[0]
        neg_id = tokenizer.encode(neg_word, add_special_tokens=False)[0]

        delta_pos = float(logits_int[pos_id].cpu() - logits_base[pos_id].cpu())
        delta_neg = float(logits_int[neg_id].cpu() - logits_base[neg_id].cpu())
        delta_diff = delta_pos - delta_neg

        single_interventions[name] = {
            "delta_pos": delta_pos,
            "delta_neg": delta_neg,
            "delta_diff": delta_diff,
        }
        print(f"  {name}: Δ_pos={delta_pos:.3f}, Δ_neg={delta_neg:.3f}, Δ_diff={delta_diff:.3f}")

    # 5. 双维度和三维度联合干预
    print(f"\n  多维度联合干预:")
    multi_interventions = {}

    import itertools
    for r in [2, 3]:
        for combo in itertools.combinations(dim_names, r):
            combo_key = "+".join(combo)
            direction_sum = np.zeros_like(dim_directions[dim_names[0]]["direction"])
            for name in combo:
                direction_sum += dim_directions[name]["direction"]
            direction_sum_norm = direction_sum / np.linalg.norm(direction_sum)

            w_tensor = torch.tensor(direction_sum_norm, dtype=torch.float32, device=device)
            inputs_embeds_int = embed(tokenizer(PROMPT, return_tensors="pt").to(device).input_ids).detach().clone().to(model.dtype)
            inputs_embeds_int[0, -1, :] += (beta * w_tensor).to(model.dtype)

            position_ids = torch.arange(inputs_embeds_int.shape[1], device=device).unsqueeze(0)
            with torch.no_grad():
                logits_int = model(inputs_embeds=inputs_embeds_int, position_ids=position_ids).logits[0, -1, :]

            # 测量每个维度上的效果
            combo_results = {}
            for name in combo:
                pos_word = dim_directions[name]["pos_word"]
                neg_word = dim_directions[name]["neg_word"]
                pos_id = tokenizer.encode(pos_word, add_special_tokens=False)[0]
                neg_id = tokenizer.encode(neg_word, add_special_tokens=False)[0]

                delta_pos = float(logits_int[pos_id].cpu() - logits_base[pos_id].cpu())
                delta_neg = float(logits_int[neg_id].cpu() - logits_base[neg_id].cpu())
                combo_results[name] = {
                    "delta_pos": delta_pos,
                    "delta_neg": delta_neg,
                    "delta_diff": delta_pos - delta_neg,
                }

            multi_interventions[combo_key] = combo_results
            print(f"  {combo_key}: " + ", ".join(
                f"{n}: Δ_diff={v['delta_diff']:.3f}" for n, v in combo_results.items()
            ))

    # 6. 正交干预: 在不同维度上分别注入
    print(f"\n  正交独立干预(各维度分别注入β=5):")
    ortho_interventions = {}

    direction_sum = np.zeros_like(dim_directions[dim_names[0]]["direction"])
    for name in dim_names:
        direction_sum += dim_directions[name]["direction"] * beta
    direction_tensor = torch.tensor(direction_sum, dtype=torch.float32, device=device)

    inputs_embeds_int = embed(tokenizer(PROMPT, return_tensors="pt").to(device).input_ids).detach().clone().to(model.dtype)
    inputs_embeds_int[0, -1, :] += direction_tensor.to(model.dtype)

    position_ids = torch.arange(inputs_embeds_int.shape[1], device=device).unsqueeze(0)
    with torch.no_grad():
        logits_ortho = model(inputs_embeds=inputs_embeds_int, position_ids=position_ids).logits[0, -1, :]

    for name in dim_names:
        pos_word = dim_directions[name]["pos_word"]
        neg_word = dim_directions[name]["neg_word"]
        pos_id = tokenizer.encode(pos_word, add_special_tokens=False)[0]
        neg_id = tokenizer.encode(neg_word, add_special_tokens=False)[0]

        delta_pos = float(logits_ortho[pos_id].cpu() - logits_base[pos_id].cpu())
        delta_neg = float(logits_ortho[neg_id].cpu() - logits_base[neg_id].cpu())
        ortho_interventions[name] = {
            "delta_pos": delta_pos,
            "delta_neg": delta_neg,
            "delta_diff": delta_pos - delta_neg,
            "single_diff": single_interventions[name]["delta_diff"],
            "superposition_ratio": (delta_pos - delta_neg) / single_interventions[name]["delta_diff"] if abs(single_interventions[name]["delta_diff"]) > 0.01 else 0.0,
        }
        print(f"  {name}: 联合Δ_diff={delta_pos - delta_neg:.3f}, 单独Δ_diff={single_interventions[name]['delta_diff']:.3f}, "
              f"叠加比={ortho_interventions[name]['superposition_ratio']:.3f}")

    results = {
        "orthogonality": orthogonality,
        "extended_orthogonality": ext_orthogonality,
        "single_interventions": single_interventions,
        "multi_interventions": {k: v for k, v in multi_interventions.items()},
        "ortho_interventions": ortho_interventions,
        "dim_directions_norms": {n: {"norm": v["norm"], "pos": v["pos_word"], "neg": v["neg_word"]} for n, v in dim_directions.items()},
    }

    return results


# ========== P381: 注意力头的路径选择功能分析 ==========

def run_p381(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P381: 注意力头的路径选择功能分析 - {model_name}")
    print(f"{'='*60}")

    results = {}
    embed = model.get_input_embeddings()

    # 选择几个关键属性
    attrs = ["red", "true", "formal"]
    n_layers_to_analyze = 6  # 分析L0-L5

    layers = get_layers(model, n_layers_to_analyze)

    for attr in attrs:
        w_lm_norm_np, w_lm_attr_np = get_w_lm_normed(model, tokenizer, attr)
        w_lm_tensor = torch.tensor(w_lm_norm_np, dtype=torch.float32, device=device)

        toks = tokenizer(PROMPT, return_tensors="pt").to(device)
        input_ids = toks.input_ids
        seq_len = input_ids.shape[1]
        inputs_embeds = embed(input_ids).detach().clone().to(model.dtype)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        # 基线前向传播，捕获每层的attention输出
        captured_attn = {}
        captured_mlp = {}
        captured_residual = {}

        def make_attn_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    captured_attn[layer_idx] = output[0].detach().float()
                else:
                    captured_attn[layer_idx] = output.detach().float()
            return hook

        def make_residual_hook(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    captured_residual[layer_idx] = output[0].detach().float()
                else:
                    captured_residual[layer_idx] = output.detach().float()
            return hook

        # 注册hook
        handles = []
        for i, layer in enumerate(layers):
            # Attention输出hook
            if hasattr(layer, "self_attn"):
                h = layer.self_attn.register_forward_hook(make_attn_hook(i))
                handles.append(h)

        # 基线前向
        with torch.no_grad():
            _ = model(inputs_embeds=inputs_embeds, position_ids=position_ids)

        # 移除hook
        for h in handles:
            h.remove()

        # 干预前向(注入W_lm方向)
        inputs_embeds_int = inputs_embeds.clone()
        inputs_embeds_int[0, -1, :] += (8.0 * w_lm_tensor).to(model.dtype)

        captured_attn_int = {}

        def make_attn_hook_int(layer_idx):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    captured_attn_int[layer_idx] = output[0].detach().float()
                else:
                    captured_attn_int[layer_idx] = output.detach().float()
            return hook

        handles_int = []
        for i, layer in enumerate(layers):
            if hasattr(layer, "self_attn"):
                h = layer.self_attn.register_forward_hook(make_attn_hook_int(i))
                handles_int.append(h)

        with torch.no_grad():
            _ = model(inputs_embeds=inputs_embeds_int, position_ids=position_ids)

        for h in handles_int:
            h.remove()

        # 分析每层attention的变化
        layer_analysis = {}
        for i in range(n_layers_to_analyze):
            if i not in captured_attn or i not in captured_attn_int:
                continue

            attn_base = captured_attn[i][0, -1, :].cpu().numpy()
            attn_int = captured_attn_int[i][0, -1, :].cpu().numpy()
            delta_attn = attn_int - attn_base

            # 投影到W_lm方向
            proj_wlm = float(np.dot(delta_attn, w_lm_norm_np))
            delta_norm = float(np.linalg.norm(delta_attn))
            cos_wlm = proj_wlm / delta_norm if delta_norm > 1e-8 else 0.0

            layer_analysis[f"L{i}"] = {
                "attn_delta_norm": delta_norm,
                "attn_proj_wlm": proj_wlm,
                "attn_cos_wlm": cos_wlm,
            }
            print(f"  {attr} L{i}: ||Δattn||={delta_norm:.4f}, proj_wlm={proj_wlm:.4f}, cos_wlm={cos_wlm:.4f}")

        # 分析单个注意力头的贡献(仅L0)
        layer0 = layers[0]
        n_heads = None
        head_dim = None

        # 获取注意力头的数量和维度
        if hasattr(layer0, "self_attn") and hasattr(layer0.self_attn, "num_heads"):
            n_heads = layer0.self_attn.num_heads
            head_dim = layer0.self_attn.head_dim if hasattr(layer0.self_attn, "head_dim") else None
        elif hasattr(layer0, "self_attn") and hasattr(layer0.self_attn, "config"):
            n_heads = getattr(layer0.self_attn.config, "num_attention_heads", None)
            head_dim = getattr(layer0.self_attn.config, "head_dim", None)

        if n_heads is None:
            # 尝试从权重推断
            if hasattr(layer0, "self_attn"):
                if hasattr(layer0.self_attn, "q_proj"):
                    q_weight = layer0.self_attn.q_proj.weight
                    hidden_dim = q_weight.shape[1]
                    # 尝试常见的头数
                    for nh in [32, 16, 8, 12, 20, 40, 48]:
                        if hidden_dim % nh == 0:
                            n_heads = nh
                            head_dim = hidden_dim // nh
                            break

        head_analysis = {}
        if n_heads is not None and 0 in captured_attn and 0 in captured_attn_int:
            attn_base = captured_attn[0][0, -1, :].cpu()  # [hidden_dim]
            attn_int = captured_attn_int[0][0, -1, :].cpu()
            delta_attn = attn_int - attn_base
            hidden_dim = attn_base.shape[0]

            if head_dim is not None and hidden_dim == n_heads * head_dim:
                print(f"\n  Attention head analysis (L0, {n_heads} heads, head_dim={head_dim}):")
                delta_attn_reshaped = delta_attn.view(n_heads, head_dim)
                w_lm_reshaped = torch.tensor(w_lm_norm_np, dtype=torch.float32).view(n_heads, head_dim)

                for h in range(min(n_heads, 8)):
                    delta_h = delta_attn_reshaped[h].numpy()
                    wlm_h = w_lm_reshaped[h].numpy()
                    proj_h = float(np.dot(delta_h, wlm_h))
                    norm_h = float(np.linalg.norm(delta_h))

                    head_analysis[f"head_{h}"] = {
                        "delta_norm": norm_h,
                        "proj_wlm": proj_h,
                    }
                    if norm_h > 0.01:
                        print(f"    head_{h}: ||delta||={norm_h:.4f}, proj={proj_h:.4f}")
            else:
                nh_hd = n_heads * head_dim if head_dim is not None else "unknown"
                print(f"    hidden_dim={hidden_dim} != n_heads*head_dim={nh_hd}, using overall analysis")
                delta_np = delta_attn.numpy()
                proj = float(np.dot(delta_np, w_lm_norm_np))
                norm = float(np.linalg.norm(delta_np))
                head_analysis["overall"] = {"delta_norm": norm, "proj_wlm": proj}

        results[attr] = {
            "layer_analysis": layer_analysis,
            "head_analysis": head_analysis,
            "n_heads": n_heads,
            "head_dim": head_dim,
        }

    return results


# ========== P382: 残差流的路径记忆量化 ==========

def run_p382(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P382: 残差流的路径记忆量化 - {model_name}")
    print(f"{'='*60}")

    results = {}
    embed = model.get_input_embeddings()
    attr = "red"

    w_lm_norm_np, _ = get_w_lm_normed(model, tokenizer, attr)
    w_lm_tensor = torch.tensor(w_lm_norm_np, dtype=torch.float32, device=device)

    toks = tokenizer(PROMPT, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    seq_len = input_ids.shape[1]
    inputs_embeds_base = embed(input_ids).detach().clone().to(model.dtype)

    # 干预版本
    inputs_embeds_int = inputs_embeds_base.clone()
    inputs_embeds_int[0, -1, :] += (8.0 * w_lm_tensor).to(model.dtype)

    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    n_layers_total = len(get_layers(model))
    n_scan = min(n_layers_total, 10)  # 扫描前10层
    layers = get_layers(model, n_scan)

    # 捕获每层的residual stream, attention输出, MLP输出
    captured = {
        "base": {},  # 基线的residual
        "int": {},   # 干预的residual
    }

    def make_hook(storage, key):
        def hook(module, input, output):
            if isinstance(output, tuple):
                storage[key] = output[0].detach().float()
            else:
                storage[key] = output.detach().float()
        return hook

    # 基线前向
    handles = []
    for i, layer in enumerate(layers):
        h = layer.register_forward_hook(make_hook(captured["base"], f"L{i}"))
        handles.append(h)

    with torch.no_grad():
        _ = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids)

    for h in handles:
        h.remove()

    # 干预前向
    handles_int = []
    for i, layer in enumerate(layers):
        h = layer.register_forward_hook(make_hook(captured["int"], f"L{i}"))
        handles_int.append(h)

    with torch.no_grad():
        _ = model(inputs_embeds=inputs_embeds_int, position_ids=position_ids)

    for h in handles_int:
        h.remove()

    # 分析每层
    layer_data = []
    for i in range(n_scan):
        key = f"L{i}"
        if key not in captured["base"] or key not in captured["int"]:
            continue

        h_base = captured["base"][key][0, -1, :].cpu().numpy()
        h_int = captured["int"][key][0, -1, :].cpu().numpy()
        delta_h = h_int - h_base

        # 总Δ在W_lm方向的投影
        proj_wlm = float(np.dot(delta_h, w_lm_norm_np))
        delta_norm = float(np.linalg.norm(delta_h))
        cos_wlm = proj_wlm / delta_norm if delta_norm > 1e-8 else 0.0

        # 上一层的Δ (residual记忆)
        if i == 0:
            proj_residual = proj_wlm  # L0的"残差记忆"就是注入本身
            cos_residual = cos_wlm
        else:
            prev_key = f"L{i-1}"
            if prev_key in captured["base"] and prev_key in captured["int"]:
                h_base_prev = captured["base"][prev_key][0, -1, :].cpu().numpy()
                h_int_prev = captured["int"][prev_key][0, -1, :].cpu().numpy()
                delta_prev = h_int_prev - h_base_prev
                proj_residual = float(np.dot(delta_prev, w_lm_norm_np))
                cos_residual = float(np.dot(delta_prev, delta_h) / (np.linalg.norm(delta_prev) * delta_norm)) if delta_norm > 1e-8 and np.linalg.norm(delta_prev) > 1e-8 else 0.0
            else:
                proj_residual = 0.0
                cos_residual = 0.0

        # MLP+Attention的更新量 = 当前层Δ - 上一层Δ (残差连接使得这部分可分离)
        delta_update = delta_h - (h_int_prev - h_base_prev) if i > 0 and prev_key in captured["base"] else delta_h
        update_norm = float(np.linalg.norm(delta_update))
        proj_update = float(np.dot(delta_update, w_lm_norm_np)) if update_norm > 1e-8 else 0.0

        # 路径记忆比例 = 残差投影 / 总投影
        memory_ratio = abs(proj_residual) / abs(proj_wlm) if abs(proj_wlm) > 1e-8 else 0.0

        r = {
            "layer": i,
            "cos_wlm": cos_wlm,
            "proj_wlm": proj_wlm,
            "delta_norm": delta_norm,
            "proj_residual": proj_residual,
            "cos_residual": cos_residual,
            "proj_update": proj_update,
            "update_norm": update_norm,
            "memory_ratio": memory_ratio,
        }
        layer_data.append(r)
        print(f"  L{i}: cos={cos_wlm:.4f}, proj_wlm={proj_wlm:.4f}, "
              f"残差记忆={proj_residual:.4f}, 更新={proj_update:.4f}, "
              f"记忆比={memory_ratio:.2%}")

    results = {
        "attr": attr,
        "layer_data": layer_data,
        "n_layers_scanned": n_scan,
        "total_layers": n_layers_total,
    }

    return results


# ========== P383: 动态路径选择的数学模型拟合 ==========

def run_p383(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P383: 动态路径选择数学模型拟合 - {model_name}")
    print(f"{'='*60}")

    results = {}
    embed = model.get_input_embeddings()
    attrs = ["red", "blue"]
    beta = 8.0

    n_layers_total = len(get_layers(model))
    n_scan = min(n_layers_total, 16)
    layers = get_layers(model, n_scan)

    for attr in attrs:
        w_lm_norm_np, _ = get_w_lm_normed(model, tokenizer, attr)
        w_lm_tensor = torch.tensor(w_lm_norm_np, dtype=torch.float32, device=device)

        toks = tokenizer(PROMPT, return_tensors="pt").to(device)
        input_ids = toks.input_ids
        seq_len = input_ids.shape[1]
        inputs_embeds_base = embed(input_ids).detach().clone().to(model.dtype)
        inputs_embeds_int = inputs_embeds_base.clone()
        inputs_embeds_int[0, -1, :] += (beta * w_lm_tensor).to(model.dtype)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        # 捕获每层输出
        captured_base = {}
        captured_int = {}

        def make_hook(storage, key):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    storage[key] = output[0].detach().float()
                else:
                    storage[key] = output.detach().float()
            return hook

        handles_b = []
        handles_i = []
        for i, layer in enumerate(layers):
            handles_b.append(layer.register_forward_hook(make_hook(captured_base, f"L{i}")))
            handles_i.append(layer.register_forward_hook(make_hook(captured_int, f"L{i}")))

        with torch.no_grad():
            _ = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids)
        for h in handles_b:
            h.remove()

        with torch.no_grad():
            _ = model(inputs_embeds=inputs_embeds_int, position_ids=position_ids)
        for h in handles_i:
            h.remove()

        # 计算每层的cos曲线
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

        # 获取每层的W_up谱范数
        w_up_spectrum = []
        for i, layer in enumerate(layers):
            mlp = get_mlp(model, layer)
            if hasattr(mlp, "up_proj"):
                W = mlp.up_proj.weight.detach().cpu().float()
            elif hasattr(mlp, "gate_up_proj"):
                W_full = mlp.gate_up_proj.weight.detach().cpu().float()
                inter_size = W_full.shape[0] // 2
                W = W_full[inter_size:, :]
            else:
                w_up_spectrum.append(0.0)
                continue

            try:
                _, S, _ = torch.linalg.svd(W, full_matrices=False)
                w_up_spectrum.append(float(S[0].item()))
            except:
                w_up_spectrum.append(float(W.norm().item()))

        # 拟合模型: cos(t) = cos(0) * exp(-λ * Σ σ_up(k))
        # 其中Σ σ_up(k)是累积W_up谱范数
        if len(cos_curve) >= 3:
            cos0 = cos_curve[0]["cos"]
            if cos0 < 0.01:
                cos0 = 0.99  # fallback

            # 累积W_up范数
            cum_sigma_up = np.cumsum(w_up_spectrum)

            # 对数拟合: ln(cos(t)/cos(0)) = -λ * cum_sigma_up(t)
            lam_values = []
            for i in range(1, min(len(cos_curve), len(cum_sigma_up))):
                if cos_curve[i]["cos"] > 0.01 and cum_sigma_up[i] > 0:
                    lam = -np.log(cos_curve[i]["cos"] / cos0) / cum_sigma_up[i]
                    lam_values.append({"layer": i, "lambda": lam, "cum_sigma_up": float(cum_sigma_up[i])})

            # 平均λ
            if lam_values:
                avg_lambda = np.mean([v["lambda"] for v in lam_values if v["lambda"] > 0])
                # 预测半衰期
                if avg_lambda > 0:
                    # 半衰期: cos(half_life) = cos(0)/2 → Σ σ_up = ln(2)/λ
                    half_life_cum_sigma = np.log(2) / avg_lambda
                    # 找到达到这个累积σ的层数
                    half_life_layers = 0
                    cum = 0
                    for s in w_up_spectrum:
                        cum += s
                        half_life_layers += 1
                        if cum >= half_life_cum_sigma:
                            break
                else:
                    half_life_layers = n_scan
                    half_life_cum_sigma = 0
                    avg_lambda = 0

                print(f"  {attr}: cos(0)={cos0:.4f}, λ={avg_lambda:.4f}, "
                      f"半衰期={half_life_layers}层 (累积σ={half_life_cum_sigma:.2f})")
            else:
                avg_lambda = 0
                half_life_layers = n_scan

            # 指数模型预测 vs 实际
            predictions = []
            for i in range(min(len(cos_curve), len(cum_sigma_up))):
                if avg_lambda > 0:
                    pred_cos = cos0 * np.exp(-avg_lambda * cum_sigma_up[i])
                else:
                    pred_cos = cos0
                actual_cos = cos_curve[i]["cos"]
                predictions.append({
                    "layer": i,
                    "actual_cos": actual_cos,
                    "predicted_cos": float(pred_cos),
                    "error": abs(actual_cos - pred_cos),
                })

            print(f"\n  指数衰减模型验证:")
            for p in predictions[:8]:
                print(f"    L{p['layer']}: actual={p['actual_cos']:.4f}, pred={p['predicted_cos']:.4f}, err={p['error']:.4f}")

            results[attr] = {
                "cos_curve": cos_curve,
                "w_up_spectrum": w_up_spectrum,
                "avg_lambda": float(avg_lambda),
                "half_life_layers": half_life_layers,
                "predictions": predictions,
                "lambda_values": lam_values,
            }
        else:
            results[attr] = {"cos_curve": cos_curve, "error": "insufficient data"}

    return results


# ========== 主函数 ==========

def main():
    parser = argparse.ArgumentParser(description="Phase LXXIV: Dynamic Path Selection Verification")
    parser.add_argument("--model", type=str, default="qwen3", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=str, default="all", choices=["p380", "p381", "p382", "p383", "all"])
    args = parser.parse_args()

    models_to_run = ["qwen3", "glm4", "deepseek7b"] if args.model == "qwen3" else [args.model]

    for model_name in models_to_run:
        print(f"\n{'#'*70}")
        print(f"# 开始测试模型: {model_name}")
        print(f"{'#'*70}")

        model, tokenizer, device = load_model(model_name)
        timestamp = time.strftime("%Y%m%d_%H%M")

        all_results = {"model": model_name, "timestamp": timestamp}

        try:
            if args.exp in ["p380", "all"]:
                r380 = run_p380(model, tokenizer, device, model_name)
                all_results["p380"] = r380

            if args.exp in ["p381", "all"]:
                r381 = run_p381(model, tokenizer, device, model_name)
                all_results["p381"] = r381

            if args.exp in ["p382", "all"]:
                r382 = run_p382(model, tokenizer, device, model_name)
                all_results["p382"] = r382

            if args.exp in ["p383", "all"]:
                r383 = run_p383(model, tokenizer, device, model_name)
                all_results["p383"] = r383

        except Exception as e:
            print(f"  ERROR in {model_name}: {e}")
            traceback.print_exc()
            all_results["error"] = str(e)

        # 保存结果
        out_file = OUT_DIR / f"phase_lxxiv_p380_383_{model_name}_{timestamp}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n结果已保存: {out_file}")

        # 释放GPU内存
        del model
        torch.cuda.empty_cache()
        print(f"GPU内存已释放，等待5秒...")
        time.sleep(5)


if __name__ == "__main__":
    main()
