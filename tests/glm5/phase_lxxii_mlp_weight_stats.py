"""
Phase LXXII-P374/375/376: MLP权重统计与SiLU工作点分析
======================================================================

核心目标: 理解为什么不同模型的MLP Jacobian范数差异如此巨大(0.0003 vs 40.8)

P374: MLP权重矩阵的统计特性
  - W_down的谱范数(||W||_2 = 最大奇异值)
  - W_down的Frobenius范数(||W||_F)
  - W_down的条件数(κ = σ_max/σ_min)
  - W_up, W_gate的范数
  - 看GLM4的W_down是否具有"低增益"特性

P375: SiLU激活函数的"工作点"分析
  - 统计L0-L5每层gate输出(W_gate(x))的分布
  - 计算SiLU'(gate_out)的均值/中位数/分位数
  - GLM4的gate是否在SiLU的"低梯度区"(|z|<1)工作?
  - DS7B的gate是否在"高梯度区"(|z|>>1)工作?

P376: 跨层Jacobian衰减分析
  - 计算L0-L5每层的||J_W|| (MLP Jacobian范数)
  - 看Jacobian是逐层递增还是递减
  - Jacobian递增速率与cos衰减速率的关系
  - 构建方向保持度的精确数学模型

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

ATTRS = ["red", "blue", "big", "small", "hot", "cold"]
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


def get_w_lm_normed(model, tokenizer, attr):
    attr_tok_ids = tokenizer.encode(attr, add_special_tokens=False)
    attr_tok_id = attr_tok_ids[0]
    lm_head = model.lm_head
    w_lm_attr = lm_head.weight[attr_tok_id].detach().cpu().float()
    w_lm_norm = w_lm_attr / w_lm_attr.norm()
    return w_lm_norm.numpy(), w_lm_attr.numpy()


def get_layers(model, max_layers=6):
    if hasattr(model.model, "layers"):
        return list(model.model.layers[:max_layers])
    elif hasattr(model.model, "encoder"):
        return list(model.model.encoder.layers[:max_layers])
    raise ValueError("Cannot find layers")


def get_mlp(model, layer):
    if hasattr(layer, "mlp"):
        return layer.mlp
    elif hasattr(layer, "feed_forward"):
        return layer.feed_forward
    raise ValueError("Cannot find MLP")


# ========== P374: MLP权重统计 ==========

def run_p374(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P374: MLP权重矩阵统计特性 - {model_name}")
    print(f"{'='*60}")

    results = {}
    layers = get_layers(model, max_layers=6)

    for li, layer in enumerate(layers):
        mlp = get_mlp(model, layer)
        layer_result = {}

        # 获取权重矩阵
        weight_matrices = {}

        if hasattr(mlp, "gate_proj"):
            weight_matrices["W_gate"] = mlp.gate_proj.weight.detach().cpu().float()
        if hasattr(mlp, "up_proj"):
            weight_matrices["W_up"] = mlp.up_proj.weight.detach().cpu().float()
        if hasattr(mlp, "down_proj"):
            weight_matrices["W_down"] = mlp.down_proj.weight.detach().cpu().float()
        if hasattr(mlp, "gate_up_proj"):
            # GLM4合并结构: [2*intermediate, hidden]
            w = mlp.gate_up_proj.weight.detach().cpu().float()
            intermediate_size = w.shape[0] // 2
            weight_matrices["W_gate"] = w[:intermediate_size, :]
            weight_matrices["W_up"] = w[intermediate_size:, :]
        if hasattr(mlp, "dense_h_to_4h"):
            weight_matrices["W_up_full"] = mlp.dense_h_to_4h.weight.detach().cpu().float()
        if hasattr(mlp, "dense_4h_to_h"):
            weight_matrices["W_down_full"] = mlp.dense_4h_to_h.weight.detach().cpu().float()

        for name, W in weight_matrices.items():
            # Frobenius范数
            frobenius = float(W.norm().item())
            # 谱范数 (最大奇异值) - 只计算top1避免太贵
            try:
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                spectral = float(S[0].item())
                # 条件数 (只看top/bottom 10)
                n_sv = min(len(S), 50)
                condition = float(S[0].item() / S[n_sv-1].item()) if S[n_sv-1] > 1e-8 else float('inf')
                # 前10个奇异值的均值
                sv_mean = float(S[:10].mean().item())
                # 奇异值分布 (取前20)
                sv_top20 = S[:20].tolist()
            except Exception as e:
                spectral = 0.0
                condition = 0.0
                sv_mean = 0.0
                sv_top20 = []

            # 权重的统计
            w_mean = float(W.mean().item())
            w_std = float(W.std().item())

            layer_result[name] = {
                "shape": list(W.shape),
                "frobenius": frobenius,
                "spectral": spectral,
                "condition_top50": condition,
                "sv_mean_top10": sv_mean,
                "sv_top20": sv_top20[:10],
                "weight_mean": w_mean,
                "weight_std": w_std,
            }

            print(f"  L{li} {name}: shape={list(W.shape)}, ||W||_F={frobenius:.2f}, "
                  f"σ_max={spectral:.2f}, κ={condition:.1f}, sv_mean={sv_mean:.2f}")

        # 关键指标: W_down的"增益"
        if "W_down" in layer_result:
            dr = layer_result["W_down"]
            layer_result["down_gain"] = dr["spectral"]  # W_down的谱范数 = 最大增益

        results[f"L{li}"] = layer_result

    return results


# ========== P375: SiLU工作点分析 ==========

def run_p375(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P375: SiLU激活函数工作点分析 - {model_name}")
    print(f"{'='*60}")

    results = {}
    layers = get_layers(model, max_layers=6)
    embed = model.get_input_embeddings()

    for attr in ATTRS[:2]:  # 只测2个属性词
        w_lm_norm_np, _ = get_w_lm_normed(model, tokenizer, attr)
        w_lm_tensor = torch.tensor(w_lm_norm_np, dtype=torch.float32, device=device)

        toks = tokenizer(PROMPT, return_tensors="pt").to(device)
        input_ids = toks.input_ids
        seq_len = input_ids.shape[1]
        inputs_embeds = embed(input_ids).detach().clone().to(model.dtype)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        attr_results = {}

        for li, layer in enumerate(layers):
            mlp = get_mlp(model, layer)

            # Hook: MLP输入 (post_attention_layernorm输出)
            captured_mlp_input = {}

            def make_input_hook(key):
                def hook(module, input, output):
                    if isinstance(input, tuple):
                        captured_mlp_input[key] = input[0].detach().float()
                    else:
                        captured_mlp_input[key] = input.detach().float()
                return hook

            # 找layernorm
            if hasattr(layer, "post_attention_layernorm"):
                ln = layer.post_attention_layernorm
            elif hasattr(layer, "input_layernorm"):
                ln = layer.input_layernorm
            else:
                continue

            h = ln.register_forward_hook(make_input_hook(f"L{li}"))
            with torch.no_grad():
                _ = model(inputs_embeds=inputs_embeds, position_ids=position_ids)
            h.remove()

            if f"L{li}" not in captured_mlp_input:
                continue

            mlp_input = captured_mlp_input[f"L{li}"][0, -1, :]  # [hidden_dim]

            # 计算gate输出
            mlp_dtype = next(mlp.parameters()).dtype
            x = mlp_input.unsqueeze(0).unsqueeze(0).to(mlp_dtype)

            with torch.no_grad():
                if hasattr(mlp, "gate_proj"):
                    gate_out = mlp.gate_proj(x)[0, 0, :].detach().float().cpu().numpy()
                elif hasattr(mlp, "gate_up_proj"):
                    w = mlp.gate_up_proj.weight.detach().cpu().float()
                    intermediate_size = w.shape[0] // 2
                    W_gate = w[:intermediate_size, :]
                    gate_out = (W_gate @ mlp_input.cpu().float()).numpy()
                else:
                    continue

            # SiLU'(z) = sigmoid(z) * (1 + z * (1 - sigmoid(z)))
            # 简化: SiLU'(z) = sigmoid(z) + z * sigmoid(z) * (1 - sigmoid(z))
            # 或直接: SiLU'(z) = sigmoid(z) * (1 - z*sigmoid(z) + z)
            sigmoid_z = 1.0 / (1.0 + np.exp(-gate_out))
            silu_grad = sigmoid_z * (1.0 + gate_out * (1.0 - sigmoid_z))

            lr = {
                "gate_mean": float(np.mean(gate_out)),
                "gate_std": float(np.std(gate_out)),
                "gate_median": float(np.median(gate_out)),
                "gate_p10": float(np.percentile(gate_out, 10)),
                "gate_p90": float(np.percentile(gate_out, 90)),
                "gate_abs_mean": float(np.mean(np.abs(gate_out))),
                "gate_neg_frac": float(np.mean(gate_out < 0)),
                "gate_small_frac": float(np.mean(np.abs(gate_out) < 1.0)),  # |z|<1
                "silu_grad_mean": float(np.mean(silu_grad)),
                "silu_grad_median": float(np.median(silu_grad)),
                "silu_grad_p10": float(np.percentile(silu_grad, 10)),
                "silu_grad_p90": float(np.percentile(silu_grad, 90)),
                "silu_grad_small_frac": float(np.mean(silu_grad < 0.5)),  # 梯度<0.5
            }
            attr_results[f"L{li}"] = lr
            print(f"  {attr} L{li}: gate_mean={lr['gate_mean']:.2f}, gate_std={lr['gate_std']:.2f}, "
                  f"|gate|_mean={lr['gate_abs_mean']:.2f}, |z|<1: {lr['gate_small_frac']:.1%}, "
                  f"silu'_mean={lr['silu_grad_mean']:.3f}, silu'<0.5: {lr['silu_grad_small_frac']:.1%}")

        results[attr] = attr_results

    return results


# ========== P376: 跨层Jacobian衰减分析 ==========

def run_p376(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P376: 跨层Jacobian衰减分析 - {model_name}")
    print(f"{'='*60}")

    results = {}
    layers = get_layers(model, max_layers=6)
    embed = model.get_input_embeddings()

    for attr in ATTRS[:2]:
        w_lm_norm_np, _ = get_w_lm_normed(model, tokenizer, attr)
        w_lm_tensor = torch.tensor(w_lm_norm_np, dtype=torch.float32, device=device)

        toks = tokenizer(PROMPT, return_tensors="pt").to(device)
        input_ids = toks.input_ids
        seq_len = input_ids.shape[1]
        inputs_embeds_base = embed(input_ids).detach().clone().to(model.dtype)

        beta = 8.0
        inputs_embeds_intervened = inputs_embeds_base.clone()
        inputs_embeds_intervened[0, -1, :] += (beta * w_lm_tensor).to(model.dtype)

        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        attr_results = {}

        for li, layer in enumerate(layers):
            mlp = get_mlp(model, layer)

            # Hook: 获取MLP输入
            captured_mlp_input_base = {}
            captured_mlp_input_interv = {}

            def make_base_hook(key):
                def hook(module, input, output):
                    if isinstance(input, tuple):
                        captured_mlp_input_base[key] = input[0].detach().float()
                    else:
                        captured_mlp_input_base[key] = input.detach().float()
                return hook

            def make_interv_hook(key):
                def hook(module, input, output):
                    if isinstance(input, tuple):
                        captured_mlp_input_interv[key] = input[0].detach().float()
                    else:
                        captured_mlp_input_interv[key] = input.detach().float()
                return hook

            if hasattr(layer, "post_attention_layernorm"):
                ln = layer.post_attention_layernorm
            elif hasattr(layer, "input_layernorm"):
                ln = layer.input_layernorm
            else:
                continue

            # Base forward
            h1 = ln.register_forward_hook(make_base_hook(f"L{li}"))
            with torch.no_grad():
                _ = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids)
            h1.remove()

            # Intervened forward
            h2 = ln.register_forward_hook(make_interv_hook(f"L{li}"))
            with torch.no_grad():
                _ = model(inputs_embeds=inputs_embeds_intervened, position_ids=position_ids)
            h2.remove()

            if f"L{li}" not in captured_mlp_input_base or f"L{li}" not in captured_mlp_input_interv:
                continue

            mlp_input_base = captured_mlp_input_base[f"L{li}"][0, -1, :]
            mlp_input_interv = captured_mlp_input_interv[f"L{li}"][0, -1, :]

            # 计算Jacobian范数 (有限差分)
            eps = 1e-3
            mlp_dtype = next(mlp.parameters()).dtype

            x_base = mlp_input_base.unsqueeze(0).unsqueeze(0).to(mlp_dtype)
            x_plus = x_base + eps * w_lm_tensor.to(mlp_dtype).unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                try:
                    mlp_out_base = mlp(x_base)
                    if isinstance(mlp_out_base, tuple):
                        mlp_out_base = mlp_out_base[0]
                    mlp_out_plus = mlp(x_plus)
                    if isinstance(mlp_out_plus, tuple):
                        mlp_out_plus = mlp_out_plus[0]
                except:
                    continue

            J_W = (mlp_out_plus - mlp_out_base) / eps
            J_W_vec = J_W[0, 0, :].detach().float().cpu().numpy()
            J_W_norm = float(np.linalg.norm(J_W_vec))

            # MLP输出的Δ范数
            mlp_base_out = mlp_out_base[0, 0, :].detach().float().cpu().numpy()
            mlp_interv_out = mlp_out_plus[0, 0, :].detach().float().cpu().numpy()
            # 实际上mlp_out_plus是用x+eps*W_lm的输入,不是intervened输入
            # 需要分别用base和intervened的MLP输入
            x_interv = mlp_input_interv.unsqueeze(0).unsqueeze(0).to(mlp_dtype)
            with torch.no_grad():
                mlp_out_interv = mlp(x_interv)
                if isinstance(mlp_out_interv, tuple):
                    mlp_out_interv = mlp_out_interv[0]
            mlp_base_np = mlp_out_base[0, 0, :].detach().float().cpu().numpy()
            mlp_interv_np = mlp_out_interv[0, 0, :].detach().float().cpu().numpy()
            delta_mlp_norm = float(np.linalg.norm(mlp_interv_np - mlp_base_np))

            # 层输出的cos和proj
            # Hook层输出
            captured_layer_base = {}
            captured_layer_interv = {}

            def make_lbase_hook(key):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        captured_layer_base[key] = output[0].detach().float()
                    else:
                        captured_layer_base[key] = output.detach().float()
                return hook

            def make_linterv_hook(key):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        captured_layer_interv[key] = output[0].detach().float()
                    else:
                        captured_layer_interv[key] = output.detach().float()
                return hook

            h3 = layer.register_forward_hook(make_lbase_hook(f"L{li}"))
            with torch.no_grad():
                _ = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids)
            h3.remove()

            h4 = layer.register_forward_hook(make_linterv_hook(f"L{li}"))
            with torch.no_grad():
                _ = model(inputs_embeds=inputs_embeds_intervened, position_ids=position_ids)
            h4.remove()

            if f"L{li}" not in captured_layer_base or f"L{li}" not in captured_layer_interv:
                continue

            h_base_np = captured_layer_base[f"L{li}"][0, -1, :].cpu().numpy()
            h_interv_np = captured_layer_interv[f"L{li}"][0, -1, :].cpu().numpy()
            delta_h = h_interv_np - h_base_np
            delta_h_norm = float(np.linalg.norm(delta_h))
            cos_wlm = float(np.dot(delta_h, w_lm_norm_np) / delta_h_norm) if delta_h_norm > 1e-8 else 0.0
            proj_wlm = float(np.dot(delta_h, w_lm_norm_np))

            lr = {
                "J_W_norm": J_W_norm,
                "delta_mlp_norm": delta_mlp_norm,
                "cos_wlm": cos_wlm,
                "proj_wlm": proj_wlm,
                "delta_h_norm": delta_h_norm,
                "interference": delta_mlp_norm / (proj_wlm + 1e-8) if proj_wlm > 0 else float('inf'),
            }
            attr_results[f"L{li}"] = lr
            print(f"  {attr} L{li}: ||J_W||={J_W_norm:.4f}, ||ΔMLP||={delta_mlp_norm:.2f}, "
                  f"cos={cos_wlm:.4f}, proj={proj_wlm:.2f}, interf={lr['interference']:.1%}")

        results[attr] = attr_results

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--experiment", type=str, default="all", choices=["all", "p374", "p375", "p376"])
    args = parser.parse_args()

    model_name = args.model
    timestamp = time.strftime("%Y%m%d_%H%M")

    model, tokenizer, device = load_model(model_name)
    all_results = {"model": model_name, "timestamp": timestamp}

    try:
        if args.experiment in ["all", "p374"]:
            all_results["p374"] = run_p374(model, tokenizer, device, model_name)
        if args.experiment in ["all", "p375"]:
            all_results["p375"] = run_p375(model, tokenizer, device, model_name)
        if args.experiment in ["all", "p376"]:
            all_results["p376"] = run_p376(model, tokenizer, device, model_name)
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        out_file = OUT_DIR / f"phase_lxxii_p374_376_{model_name}_{timestamp}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {out_file}")
        del model
        import gc; gc.collect(); torch.cuda.empty_cache()
        print("Model unloaded")


if __name__ == "__main__":
    main()
