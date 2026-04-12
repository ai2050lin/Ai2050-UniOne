"""
Phase LXXVIII-P393/394/395: 新信号指标 + 激活修补 + 维度竞争因果机制
=========================================================================

阶段B深化任务 - 解决cos与dlogit的矛盾:

P393: 新信号传播效率指标 cos(dh, W_target) ★★★最关键★★★
  - P390发现: cos(h_int, h_base)与delta_logit矛盾
  - 核心假设: 真正的信号传播效率应该是 cos(Δh, W_target_direction)
    其中W_target_direction是目标词的W_lm方向
  - 如果信号被精确旋转到目标方向: cos(Δh, W_target) ≈ 1 → Δlogit大
  - 如果信号被旋转到无关方向: cos(Δh, W_target) ≈ 0 → Δlogit小
  - 方法:
    1. 在L0注入维度方向
    2. 逐层捕获Δh = h_int - h_base
    3. 计算cos(Δh, W_target_pos) 和 cos(Δh, W_target_neg)
    4. 建立cos(Δh, W_target) → Δlogit 的回归
  - 预期: 这个新指标应该完美预测Δlogit!

P394: 激活修补(Activation Patching) — 找到关键计算层
  - 对每个维度, 找到"关键层": 在该层修补可以最有效地改变输出
  - 方法:
    1. 运行两个prompt(基线+注入), 收集所有层hidden state
    2. 对每层, 用注入prompt的hidden state替换基线prompt的
    3. 测量替换后输出的变化
    4. 变化最大的层 = 该维度的"关键计算层"

P395: 维度竞争的因果机制 — 逐层消融实验
  - P392发现: GLM4 logic-grammar交互=-7.38, 严重竞争
  - 目标: 确定竞争发生在哪一层
  - 方法:
    1. 同时注入logic+grammar方向
    2. 逐层检测: logic方向的Δh是否被grammar方向的Δh影响
    3. 竞争最严重的层 = "竞争层"

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

DIM_PAIRS = {
    "style": [("formal", "informal")],
    "logic": [("true", "false")],
    "grammar": [("active", "passive")],
    "sentiment": [("happy", "sad")],
    "tense": [("was", "is")],
    "certainty": [("definitely", "maybe")],
}

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


def get_w_lm(model, tokenizer, word):
    tok_ids = tokenizer.encode(word, add_special_tokens=False)
    tok_id = tok_ids[0]
    lm_head = model.lm_head
    w = lm_head.weight[tok_id].detach().cpu().float()
    w_norm = w / w.norm()
    return w_norm.numpy(), w.numpy()


def get_dimension_direction(model, tokenizer, word_pos, word_neg):
    w_pos, _ = get_w_lm(model, tokenizer, word_pos)
    w_neg, _ = get_w_lm(model, tokenizer, word_neg)
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


# ========== P393: 新信号传播效率指标 ==========

def run_p393(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P393: New signal metric cos(dh, W_target) - {model_name}")
    print(f"{'='*60}")

    embed = model.get_input_embeddings()
    test_dims = ["style", "logic", "grammar", "sentiment", "tense", "certainty"]
    beta = 8.0

    n_layers_total = len(get_layers(model))
    scan_layers = list(range(0, n_layers_total, 2))
    n_scan = max(scan_layers) + 1
    layers = get_layers(model, n_scan)

    # 维度方向 + 目标词W_lm
    dim_info = {}
    for name in test_dims:
        pos, neg = DIM_PAIRS[name][0]
        direction, norm = get_dimension_direction(model, tokenizer, pos, neg)
        w_pos, _ = get_w_lm(model, tokenizer, pos)
        w_neg, _ = get_w_lm(model, tokenizer, neg)
        dim_info[name] = {
            "direction": direction, "norm": norm,
            "pos": pos, "neg": neg,
            "w_pos_normed": w_pos, "w_neg_normed": w_neg,
        }

    toks = tokenizer(PROMPT, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    seq_len = input_ids.shape[1]
    inputs_embeds_base = embed(input_ids).detach().clone().to(model.dtype)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    # 基线logits
    with torch.no_grad():
        logits_base = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids).logits[0, -1, :].float()

    # 基线hidden states
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

    # 对每个维度在L0注入, 跟踪新指标
    new_metric_results = {}

    for dim_name in dim_info:
        direction = dim_info[dim_name]["direction"]
        w_pos = dim_info[dim_name]["w_pos_normed"]
        w_neg = dim_info[dim_name]["w_neg_normed"]
        pos_word = dim_info[dim_name]["pos"]
        neg_word = dim_info[dim_name]["neg"]
        pos_id = tokenizer.encode(pos_word, add_special_tokens=False)[0]
        neg_id = tokenizer.encode(neg_word, add_special_tokens=False)[0]

        # L0注入
        w_tensor = torch.tensor(direction * beta, dtype=torch.float32, device=device)
        inputs_embeds_int = inputs_embeds_base.clone()
        inputs_embeds_int[0, -1, :] += w_tensor.to(model.dtype)

        # 捕获注入后各层hidden state
        captured_int = {}
        handles_int = []
        for i, layer in enumerate(layers):
            if i in scan_layers:
                handles_int.append(layer.register_forward_hook(make_hook(captured_int, f"L{i}")))

        with torch.no_grad():
            logits_int = model(inputs_embeds=inputs_embeds_int, position_ids=position_ids).logits[0, -1, :].float()
        for h in handles_int:
            h.remove()

        # 逐层分析
        layer_metrics = []
        for l in scan_layers:
            key = f"L{l}"
            if key not in captured_int or key not in captured_base:
                continue

            dh = captured_int[key][0, -1, :].cpu().numpy() - captured_base[key][0, -1, :].cpu().numpy()
            dh_norm = np.linalg.norm(dh)

            # 旧指标: cos(h_int, h_base)
            h_int = captured_int[key][0, -1, :].cpu().numpy()
            h_base = captured_base[key][0, -1, :].cpu().numpy()
            norm_int = np.linalg.norm(h_int)
            norm_base = np.linalg.norm(h_base)
            cos_h = float(np.dot(h_int, h_base) / (norm_int * norm_base)) if norm_int > 1e-8 and norm_base > 1e-8 else 0

            # 新指标1: cos(dh, W_pos_direction) — 信号在目标词正方向的投影
            cos_dh_wpos = float(np.dot(dh, w_pos) / dh_norm) if dh_norm > 1e-8 else 0
            # 新指标2: cos(dh, W_neg_direction) — 信号在目标词负方向的投影
            cos_dh_wneg = float(np.dot(dh, w_neg) / dh_norm) if dh_norm > 1e-8 else 0
            # 新指标3: cos(dh, W_dim_direction) — 信号在维度方向的投影
            cos_dh_wdim = float(np.dot(dh / dh_norm, direction)) if dh_norm > 1e-8 else 0
            # 新指标4: Δh在W_pos-W_neg方向上的投影 (直接正比于Δlogit)
            w_diff = w_pos - w_neg
            w_diff_norm = np.linalg.norm(w_diff)
            proj_dh_on_wdiff = float(np.dot(dh, w_diff) / w_diff_norm) if w_diff_norm > 1e-8 else 0

            # Δh_norm
            dh_norm_val = float(dh_norm)

            layer_metrics.append({
                "layer": l,
                "cos_h_int_base": cos_h,
                "cos_dh_wpos": cos_dh_wpos,
                "cos_dh_wneg": cos_dh_wneg,
                "cos_dh_wdim": cos_dh_wdim,
                "proj_dh_on_wdiff": proj_dh_on_wdiff,
                "dh_norm": dh_norm_val,
            })

        # 最终logit变化
        delta_pos = float(logits_int[pos_id].cpu() - logits_base[pos_id].cpu())
        delta_neg = float(logits_int[neg_id].cpu() - logits_base[neg_id].cpu())
        delta_logit = delta_pos - delta_neg

        new_metric_results[dim_name] = {
            "layer_metrics": layer_metrics,
            "final_delta_logit": delta_logit,
        }

        # 关键输出: 新指标在首层和末层
        if len(layer_metrics) >= 2:
            l0 = layer_metrics[0]
            ll = layer_metrics[-1]
            print(f"  {dim_name}: L0 cos(dh,W+)={l0['cos_dh_wpos']:.4f}, cos(dh,W-)={l0['cos_dh_wneg']:.4f}, "
                  f"proj={l0['proj_dh_on_wdiff']:.3f}; "
                  f"L{ll['layer']} cos(dh,W+)={ll['cos_dh_wpos']:.4f}, proj={ll['proj_dh_on_wdiff']:.3f}; "
                  f"dlogit={delta_logit:.3f}")

    # ===== 核心分析: 哪个指标最预测delta_logit? =====
    print(f"\n  === Which metric best predicts dlogit? ===")
    # 用各层注入的per-layer intervention data (from Part 2 of P390)
    # 这里用L0注入后的逐层新指标, 与最终dlogit做回归
    # 但更好的方法: 用P390 Part2的per-layer injection数据
    # 重新计算: 在每层注入, 该层Δh的投影

    # 简化版: 用L0注入后的各层proj值, 看末层proj与dlogit的关系
    proj_vs_dlogit = []
    for dim_name in dim_info:
        metrics = new_metric_results[dim_name]["layer_metrics"]
        dlogit = new_metric_results[dim_name]["final_delta_logit"]
        # 末层的proj_on_wdiff
        if len(metrics) > 0:
            last_m = metrics[-1]
            proj_vs_dlogit.append((last_m["proj_dh_on_wdiff"], dlogit, dim_name))
            # 也添加中间层数据
            for m in metrics:
                proj_vs_dlogit.append((m["proj_dh_on_wdiff"], dlogit, dim_name))

    if len(proj_vs_dlogit) > 3:
        proj_arr = np.array([p[0] for p in proj_vs_dlogit])
        dlogit_arr = np.array([p[1] for p in proj_vs_dlogit])

        # 线性回归
        try:
            slope = np.dot(proj_arr - proj_arr.mean(), dlogit_arr - dlogit_arr.mean()) / max(np.sum((proj_arr - proj_arr.mean())**2), 1e-10)
            intercept = dlogit_arr.mean() - slope * proj_arr.mean()
            r2 = 1 - np.sum((dlogit_arr - slope * proj_arr - intercept)**2) / max(np.sum((dlogit_arr - dlogit_arr.mean())**2), 1e-10)
            print(f"  proj(dh, W_diff) vs dlogit: R2={r2:.4f}, slope={slope:.3f}")
        except:
            r2 = -999
            print(f"  Regression failed")

    results = {
        "new_metric_results": new_metric_results,
        "beta": beta,
        "test_dims": test_dims,
    }
    return results


# ========== P394: 激活修补(Activation Patching) ==========

def run_p394(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P394: Activation Patching - find key computation layers - {model_name}")
    print(f"{'='*60}")

    embed = model.get_input_embeddings()
    test_dims = ["style", "logic", "grammar", "sentiment"]
    beta = 8.0

    n_layers_total = len(get_layers(model))
    scan_layers = list(range(0, n_layers_total, 3))
    if n_layers_total - 1 not in scan_layers:
        scan_layers.append(n_layers_total - 1)
    n_scan = max(scan_layers) + 1
    layers = get_layers(model, n_scan)

    dim_info = {}
    for name in test_dims:
        pos, neg = DIM_PAIRS[name][0]
        direction, norm = get_dimension_direction(model, tokenizer, pos, neg)
        dim_info[name] = {
            "direction": direction, "norm": norm,
            "pos": pos, "neg": neg,
            "pos_id": tokenizer.encode(pos, add_special_tokens=False)[0],
            "neg_id": tokenizer.encode(neg, add_special_tokens=False)[0],
        }

    toks = tokenizer(PROMPT, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    seq_len = input_ids.shape[1]
    inputs_embeds_base = embed(input_ids).detach().clone().to(model.dtype)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    # 基线logits
    with torch.no_grad():
        logits_base = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids).logits[0, -1, :].float()

    # 注入prompt: 在L0注入
    # 收集注入后各层hidden state
    patching_results = {}

    for dim_name in dim_info:
        direction = dim_info[dim_name]["direction"]
        pos_id = dim_info[dim_name]["pos_id"]
        neg_id = dim_info[dim_name]["neg_id"]

        w_tensor = torch.tensor(direction * beta, dtype=torch.float32, device=device)
        inputs_embeds_int = inputs_embeds_base.clone()
        inputs_embeds_int[0, -1, :] += w_tensor.to(model.dtype)

        # 先收集注入后各层hidden state
        captured_int = {}
        def make_hook(storage, key):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    storage[key] = output[0].detach().clone()
                else:
                    storage[key] = output.detach().clone()
            return hook

        handles_int = []
        for i, layer in enumerate(layers):
            if i in scan_layers:
                handles_int.append(layer.register_forward_hook(make_hook(captured_int, f"L{i}")))

        with torch.no_grad():
            _ = model(inputs_embeds=inputs_embeds_int, position_ids=position_ids)
        for h in handles_int:
            h.remove()

        # 现在做激活修补: 在基线forward中, 用注入后的hidden state替换某层输出
        patch_effects = []

        for patch_layer in scan_layers:
            key = f"L{patch_layer}"
            if key not in captured_int:
                continue

            h_int_layer = captured_int[key]  # 注入后该层的hidden state

            # 在基线forward中, patch该层
            def patch_hook(module, input, output, hl=h_int_layer, pl=patch_layer):
                if isinstance(output, tuple):
                    return (hl.to(output[0].dtype),) + output[1:]
                return hl.to(output.dtype)

            handle_patch = layers[patch_layer].register_forward_hook(patch_hook)

            with torch.no_grad():
                logits_patched = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids).logits[0, -1, :].float()

            handle_patch.remove()

            delta_pos = float(logits_patched[pos_id].cpu() - logits_base[pos_id].cpu())
            delta_neg = float(logits_patched[neg_id].cpu() - logits_base[neg_id].cpu())
            patch_effect = delta_pos - delta_neg

            patch_effects.append({
                "patch_layer": patch_layer,
                "patch_effect_dlogit": patch_effect,
            })

            if patch_layer % 10 == 0 or patch_layer == scan_layers[-1]:
                print(f"  {dim_name} patch L{patch_layer}: dlogit={patch_effect:.3f}")

        patching_results[dim_name] = patch_effects

        # 找关键层
        if patch_effects:
            max_effect = max(patch_effects, key=lambda x: abs(x["patch_effect_dlogit"]))
            print(f"  {dim_name} KEY LAYER: L{max_effect['patch_layer']} (dlogit={max_effect['patch_effect_dlogit']:.3f})")

    results = {
        "patching_results": patching_results,
        "beta": beta,
        "test_dims": list(dim_info.keys()),
    }
    return results


# ========== P395: 维度竞争因果机制 ==========

def run_p395(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P395: Dimension competition causal mechanism - {model_name}")
    print(f"{'='*60}")

    embed = model.get_input_embeddings()
    # 聚焦于竞争最激烈的维度对
    test_pairs = [
        ("logic", "grammar"),  # GLM4交互=-7.38
        ("style", "grammar"),  # Qwen3交互=3.38
        ("style", "sentiment"),  # 通用
    ]
    beta = 8.0

    n_layers_total = len(get_layers(model))
    scan_layers = list(range(0, n_layers_total, 3))
    if n_layers_total - 1 not in scan_layers:
        scan_layers.append(n_layers_total - 1)
    n_scan = max(scan_layers) + 1
    layers = get_layers(model, n_scan)

    dim_info = {}
    for name in ["style", "logic", "grammar", "sentiment"]:
        pos, neg = DIM_PAIRS[name][0]
        direction, norm = get_dimension_direction(model, tokenizer, pos, neg)
        dim_info[name] = {
            "direction": direction, "norm": norm,
            "pos": pos, "neg": neg,
            "pos_id": tokenizer.encode(pos, add_special_tokens=False)[0],
            "neg_id": tokenizer.encode(neg, add_special_tokens=False)[0],
        }

    toks = tokenizer(PROMPT, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    seq_len = input_ids.shape[1]
    inputs_embeds_base = embed(input_ids).detach().clone().to(model.dtype)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    # 基线
    with torch.no_grad():
        logits_base = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids).logits[0, -1, :].float()

    # 基线hidden states
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

    competition_results = {}

    for dim1_name, dim2_name in test_pairs:
        if dim1_name not in dim_info or dim2_name not in dim_info:
            continue

        d1 = dim_info[dim1_name]
        d2 = dim_info[dim2_name]

        # 单独注入dim1
        w1 = torch.tensor(d1["direction"] * beta, dtype=torch.float32, device=device)
        ie1 = inputs_embeds_base.clone()
        ie1[0, -1, :] += w1.to(model.dtype)

        captured_d1 = {}
        handles_d1 = []
        for i, layer in enumerate(layers):
            if i in scan_layers:
                handles_d1.append(layer.register_forward_hook(make_hook(captured_d1, f"L{i}")))

        with torch.no_grad():
            logits_d1 = model(inputs_embeds=ie1, position_ids=position_ids).logits[0, -1, :].float()
        for h in handles_d1:
            h.remove()

        # 单独注入dim2
        w2 = torch.tensor(d2["direction"] * beta, dtype=torch.float32, device=device)
        ie2 = inputs_embeds_base.clone()
        ie2[0, -1, :] += w2.to(model.dtype)

        captured_d2 = {}
        handles_d2 = []
        for i, layer in enumerate(layers):
            if i in scan_layers:
                handles_d2.append(layer.register_forward_hook(make_hook(captured_d2, f"L{i}")))

        with torch.no_grad():
            logits_d2 = model(inputs_embeds=ie2, position_ids=position_ids).logits[0, -1, :].float()
        for h in handles_d2:
            h.remove()

        # 联合注入
        w12 = torch.tensor((d1["direction"] + d2["direction"]) * beta, dtype=torch.float32, device=device)
        ie12 = inputs_embeds_base.clone()
        ie12[0, -1, :] += w12.to(model.dtype)

        captured_d12 = {}
        handles_d12 = []
        for i, layer in enumerate(layers):
            if i in scan_layers:
                handles_d12.append(layer.register_forward_hook(make_hook(captured_d12, f"L{i}")))

        with torch.no_grad():
            logits_d12 = model(inputs_embeds=ie12, position_ids=position_ids).logits[0, -1, :].float()
        for h in handles_d12:
            h.remove()

        # 逐层分析竞争
        pair_key = f"{dim1_name}_vs_{dim2_name}"
        layer_competition = []

        for l in scan_layers:
            key = f"L{l}"
            if key not in captured_base or key not in captured_d1 or key not in captured_d2 or key not in captured_d12:
                continue

            h_base = captured_base[key][0, -1, :].cpu().numpy()
            dh1 = captured_d1[key][0, -1, :].cpu().numpy() - h_base
            dh2 = captured_d2[key][0, -1, :].cpu().numpy() - h_base
            dh12 = captured_d12[key][0, -1, :].cpu().numpy() - h_base

            # 线性叠加预测
            dh_linear = dh1 + dh2

            # 非线性交互 = 实际 - 线性预测
            dh_interaction = dh12 - dh_linear

            # 交互的范数
            interaction_norm = float(np.linalg.norm(dh_interaction))

            # dim1方向在联合注入中是否被dim2扭曲?
            # cos(dh12在dim1方向的投影 vs dh1在dim1方向的投影)
            d1_dir = dim_info[dim1_name]["direction"]
            d2_dir = dim_info[dim2_name]["direction"]

            proj_dh1_on_d1 = float(np.dot(dh1, d1_dir))
            proj_dh12_on_d1 = float(np.dot(dh12, d1_dir))
            proj_dh2_on_d2 = float(np.dot(dh2, d2_dir))
            proj_dh12_on_d2 = float(np.dot(dh12, d2_dir))

            # dim1的信号在联合注入时被扭曲了多少?
            proj_dh2_on_d1_dir = float(np.dot(dh2, d1_dir))
            proj_dh1_on_d2_dir = float(np.dot(dh1, d2_dir))
            dim1_distortion = proj_dh12_on_d1 - proj_dh1_on_d1 - proj_dh2_on_d1_dir  # 理论上应该是0如果线性
            dim2_distortion = proj_dh12_on_d2 - proj_dh2_on_d2 - proj_dh1_on_d2_dir

            layer_competition.append({
                "layer": l,
                "interaction_norm": interaction_norm,
                "dim1_distortion": float(dim1_distortion),
                "dim2_distortion": float(dim2_distortion),
                "proj_dh1_on_d1": proj_dh1_on_d1,
                "proj_dh12_on_d1": proj_dh12_on_d1,
                "proj_dh2_on_d2": proj_dh2_on_d2,
                "proj_dh12_on_d2": proj_dh12_on_d2,
            })

        # 最终logit竞争
        dlogit_d1_dim1 = float(logits_d1[d1["pos_id"]].cpu() - logits_d1[d1["neg_id"]].cpu() - logits_base[d1["pos_id"]].cpu() + logits_base[d1["neg_id"]].cpu())
        dlogit_d12_dim1 = float(logits_d12[d1["pos_id"]].cpu() - logits_d12[d1["neg_id"]].cpu() - logits_base[d1["pos_id"]].cpu() + logits_base[d1["neg_id"]].cpu())
        dlogit_d2_dim2 = float(logits_d2[d2["pos_id"]].cpu() - logits_d2[d2["neg_id"]].cpu() - logits_base[d2["pos_id"]].cpu() + logits_base[d2["neg_id"]].cpu())
        dlogit_d12_dim2 = float(logits_d12[d2["pos_id"]].cpu() - logits_d12[d2["neg_id"]].cpu() - logits_base[d2["pos_id"]].cpu() + logits_base[d2["neg_id"]].cpu())

        competition_results[pair_key] = {
            "layer_competition": layer_competition,
            "final_dlogit_d1_dim1": dlogit_d1_dim1,
            "final_dlogit_d12_dim1": dlogit_d12_dim1,
            "final_dlogit_d2_dim2": dlogit_d2_dim2,
            "final_dlogit_d12_dim2": dlogit_d12_dim2,
            "final_interference_dim1": dlogit_d12_dim1 - dlogit_d1_dim1,
            "final_interference_dim2": dlogit_d12_dim2 - dlogit_d2_dim2,
        }

        # 找竞争最严重的层
        if layer_competition:
            max_interact_layer = max(layer_competition, key=lambda x: abs(x["interaction_norm"]))
            max_dim1_distort = max(layer_competition, key=lambda x: abs(x["dim1_distortion"]))
            print(f"  {pair_key}: max_interaction@L{max_interact_layer['layer']} (norm={max_interact_layer['interaction_norm']:.2f}), "
                  f"max_dim1_distort@L{max_dim1_distort['layer']} ({max_dim1_distort['dim1_distortion']:.3f}), "
                  f"final_interf_d1={dlogit_d12_dim1 - dlogit_d1_dim1:.3f}, d2={dlogit_d12_dim2 - dlogit_d2_dim2:.3f}")

    results = {
        "competition_results": competition_results,
        "beta": beta,
        "test_pairs": [f"{d1}_vs_{d2}" for d1, d2 in test_pairs],
    }
    return results


# ========== Main ==========

def main():
    parser = argparse.ArgumentParser(description="Phase LXXVIII: New signal metric + activation patching + competition")
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=str, default="all", choices=["p393", "p394", "p395", "all"])
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model)
    results = {}

    timestamp = time.strftime("%Y%m%d_%H%M")

    try:
        if args.exp in ["p393", "all"]:
            r = run_p393(model, tokenizer, device, args.model)
            results["p393"] = r
    except Exception as e:
        print(f"P393 failed: {e}")
        traceback.print_exc()
        results["p393_error"] = str(e)

    try:
        if args.exp in ["p394", "all"]:
            r = run_p394(model, tokenizer, device, args.model)
            results["p394"] = r
    except Exception as e:
        print(f"P394 failed: {e}")
        traceback.print_exc()
        results["p394_error"] = str(e)

    try:
        if args.exp in ["p395", "all"]:
            r = run_p395(model, tokenizer, device, args.model)
            results["p395"] = r
    except Exception as e:
        print(f"P395 failed: {e}")
        traceback.print_exc()
        results["p395_error"] = str(e)

    out_file = OUT_DIR / f"phase_lxxviii_p393_395_{args.model}_{timestamp}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved to {out_file}")

    del model
    torch.cuda.empty_cache()
    print(f"GPU memory released for {args.model}")


if __name__ == "__main__":
    main()
