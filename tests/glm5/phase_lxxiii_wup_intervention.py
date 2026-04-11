"""
Phase LXXIII-P377/378/379: MLP权重干预与Jacobian精细分解
======================================================================

核心目标: 验证W_up谱范数与方向保持度的因果关系，精细分解Jacobian

P377: W_up缩放干预 ★★★核心因果验证★★★
  - 将Qwen3的W_up乘以缩放因子α (α从0.01到1.0)
  - 看cos是否从0.61恢复到接近0.99
  - 如果α=0.49/8.64≈0.057时cos≈0.99 → 证明W_up是因果关键!
  - 同时对GLM4做反向: 将GLM4的W_up放大到Qwen3的值

P378: Jacobian分量精细分解
  - ||J_W||² ≈ ||W_down · diag(silu') · W_up||² + ||W_down · diag(SiLU) · I||²
  - 分解为: W_down贡献 × SiLU梯度贡献 × W_up贡献
  - 看哪个因子是主要瓶颈

P379: 多层Jacobian全扫描
  - 对GLM4的48层和Qwen3的40层做完整Jacobian扫描
  - 建立J(t)的演化模型
  - 看J是否单调递增/递减/波动

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

ATTRS = ["red", "blue", "big"]
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


def compute_intervention_cos(model, tokenizer, device, attr, beta=8.0):
    """计算基础干预cos值"""
    w_lm_norm_np, _ = get_w_lm_normed(model, tokenizer, attr)
    w_lm_tensor = torch.tensor(w_lm_norm_np, dtype=torch.float32, device=device)

    toks = tokenizer(PROMPT, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    seq_len = input_ids.shape[1]
    embed = model.get_input_embeddings()
    inputs_embeds_base = embed(input_ids).detach().clone().to(model.dtype)

    inputs_embeds_intervened = inputs_embeds_base.clone()
    inputs_embeds_intervened[0, -1, :] += (beta * w_lm_tensor).to(model.dtype)

    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    # Hook L0输出
    captured_base = {}
    captured_interv = {}

    def make_hook(storage, key):
        def hook(module, input, output):
            if isinstance(output, tuple):
                storage[key] = output[0].detach().float()
            else:
                storage[key] = output.detach().float()
        return hook

    layer0 = get_layers(model, 1)[0]

    h1 = layer0.register_forward_hook(make_hook(captured_base, "L0"))
    with torch.no_grad():
        _ = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids)
    h1.remove()

    h2 = layer0.register_forward_hook(make_hook(captured_interv, "L0"))
    with torch.no_grad():
        _ = model(inputs_embeds=inputs_embeds_intervened, position_ids=position_ids)
    h2.remove()

    if "L0" not in captured_base or "L0" not in captured_interv:
        return 0.0, 0.0, 0.0

    h_base = captured_base["L0"][0, -1, :].cpu().numpy()
    h_interv = captured_interv["L0"][0, -1, :].cpu().numpy()
    delta_h = h_interv - h_base
    delta_norm = np.linalg.norm(delta_h)
    cos_val = float(np.dot(delta_h, w_lm_norm_np) / delta_norm) if delta_norm > 1e-8 else 0.0
    proj_val = float(np.dot(delta_h, w_lm_norm_np))

    return cos_val, delta_norm, proj_val


# ========== P377: W_up缩放干预 ==========

def run_p377(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P377: W_up缩放干预 - {model_name}")
    print(f"{'='*60}")

    results = {}
    layer0 = get_layers(model, 1)[0]
    mlp = get_mlp(model, layer0)

    # 获取W_up和保存原始值
    if hasattr(mlp, "up_proj"):
        W_up_original = mlp.up_proj.weight.data.clone()
        w_up_name = "up_proj"
    elif hasattr(mlp, "gate_up_proj"):
        W_combined = mlp.gate_up_proj.weight.data.clone()
        intermediate_size = W_combined.shape[0] // 2
        W_up_original = W_combined[intermediate_size:, :].clone()
        w_up_name = "gate_up_proj"
    else:
        print("  Cannot find W_up, skipping")
        return results

    # 计算W_up的谱范数
    W_up_float = W_up_original.detach().cpu().float()
    try:
        _, S_up, _ = torch.linalg.svd(W_up_float, full_matrices=False)
        sigma_max_up = float(S_up[0].item())
    except:
        sigma_max_up = float(W_up_float.norm().item())

    print(f"  W_up spectral norm: {sigma_max_up:.4f}")

    # 缩放因子列表
    alphas = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]

    for attr in ATTRS[:2]:
        attr_results = []

        for alpha in alphas:
            # 缩放W_up
            if w_up_name == "up_proj":
                mlp.up_proj.weight.data = (W_up_original * alpha).to(mlp.up_proj.weight.dtype)
            else:  # gate_up_proj
                new_combined = mlp.gate_up_proj.weight.data.clone()
                new_combined[intermediate_size:, :] = (W_up_original * alpha).to(new_combined.dtype)
                mlp.gate_up_proj.weight.data = new_combined

            # 计算cos
            cos_val, delta_norm, proj_val = compute_intervention_cos(model, tokenizer, device, attr)
            effective_sigma = sigma_max_up * alpha

            r = {
                "alpha": alpha,
                "effective_sigma_up": effective_sigma,
                "cos_wlm": cos_val,
                "delta_norm": float(delta_norm),
                "proj_wlm": float(proj_val),
            }
            attr_results.append(r)
            print(f"  {attr} α={alpha:.3f} (σ_up={effective_sigma:.2f}): cos={cos_val:.4f}, "
                  f"||Δ||={delta_norm:.2f}, proj={proj_val:.2f}")

        results[attr] = attr_results

    # 恢复原始权重
    if w_up_name == "up_proj":
        mlp.up_proj.weight.data = W_up_original
    else:
        new_combined = mlp.gate_up_proj.weight.data.clone()
        new_combined[intermediate_size:, :] = W_up_original.to(new_combined.dtype)
        mlp.gate_up_proj.weight.data = new_combined

    # 验证恢复
    cos_check, _, _ = compute_intervention_cos(model, tokenizer, device, ATTRS[0])
    print(f"  恢复后验证: cos={cos_check:.4f}")

    return results


# ========== P378: Jacobian分量分解 ==========

def run_p378(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P378: Jacobian分量精细分解 - {model_name}")
    print(f"{'='*60}")

    results = {}
    layer0 = get_layers(model, 1)[0]
    mlp = get_mlp(model, layer0)
    embed = model.get_input_embeddings()

    for attr in ATTRS[:2]:
        w_lm_norm_np, _ = get_w_lm_normed(model, tokenizer, attr)
        w_lm_tensor = torch.tensor(w_lm_norm_np, dtype=torch.float32, device=device)

        toks = tokenizer(PROMPT, return_tensors="pt").to(device)
        input_ids = toks.input_ids
        seq_len = input_ids.shape[1]
        inputs_embeds = embed(input_ids).detach().clone().to(model.dtype)

        # 获取MLP输入
        captured_mlp_input = {}

        def make_hook(key):
            def hook(module, input, output):
                if isinstance(input, tuple):
                    captured_mlp_input[key] = input[0].detach().float()
                else:
                    captured_mlp_input[key] = input.detach().float()
            return hook

        if hasattr(layer0, "post_attention_layernorm"):
            ln = layer0.post_attention_layernorm
        else:
            ln = layer0.input_layernorm

        h = ln.register_forward_hook(make_hook("mlp_in"))
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        with torch.no_grad():
            _ = model(inputs_embeds=inputs_embeds, position_ids=position_ids)
        h.remove()

        if "mlp_in" not in captured_mlp_input:
            continue

        mlp_input = captured_mlp_input["mlp_in"][0, -1, :]  # [hidden_dim]
        mlp_dtype = next(mlp.parameters()).dtype

        # 1. 分解gate和up的输出
        with torch.no_grad():
            x = mlp_input.unsqueeze(0).unsqueeze(0).to(mlp_dtype)

            if hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj"):
                gate_out = mlp.gate_proj(x)[0, 0, :].detach().float().cpu().numpy()
                up_out = mlp.up_proj(x)[0, 0, :].detach().float().cpu().numpy()
                silu_gate = 1.0 / (1.0 + np.exp(-gate_out)) * gate_out  # SiLU(z) = z*sigmoid(z)
                intermediate = silu_gate * up_out
            elif hasattr(mlp, "gate_up_proj"):
                w = mlp.gate_up_proj.weight.detach().cpu().float()
                inter_size = w.shape[0] // 2
                W_gate = w[:inter_size, :]
                W_up = w[inter_size:, :]
                x_cpu = mlp_input.cpu().float()
                gate_out = (W_gate @ x_cpu).numpy()
                up_out = (W_up @ x_cpu).numpy()
                silu_gate = 1.0 / (1.0 + np.exp(-gate_out)) * gate_out
                intermediate = silu_gate * up_out
            else:
                continue

        # 2. 计算Jacobian分量
        eps = 1e-3
        x_base = mlp_input.unsqueeze(0).unsqueeze(0).to(mlp_dtype)
        x_plus = x_base + eps * w_lm_tensor.to(mlp_dtype).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            if hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj"):
                gate_plus = mlp.gate_proj(x_plus)[0, 0, :].detach().float().cpu().numpy()
                up_plus = mlp.up_proj(x_plus)[0, 0, :].detach().float().cpu().numpy()
            elif hasattr(mlp, "gate_up_proj"):
                x_cpu_plus = (mlp_input.cpu().float() + eps * torch.tensor(w_lm_norm_np, dtype=torch.float32))
                gate_plus = (W_gate @ x_cpu_plus).numpy()
                up_plus = (W_up @ x_cpu_plus).numpy()

        # Δgate和Δup
        delta_gate = gate_plus - gate_out
        delta_up = up_plus - up_out

        # SiLU的Jacobian (对角矩阵)
        sigmoid_gate = 1.0 / (1.0 + np.exp(-gate_out))
        silu_jacobian = sigmoid_gate * (1.0 + gate_out * (1.0 - sigmoid_gate))  # SiLU'(z)

        # 3. 分解MLP Jacobian的三个路径:
        # MLP(x) = W_down(SiLU(W_gate(x)) * W_up(x))
        # ΔMLP ≈ W_down * (SiLU'(gate)*Δgate * up + SiLU(gate) * Δup)

        # 路径1: SiLU梯度路径 = silu_jacobian * delta_gate * up_out
        path1 = silu_jacobian * delta_gate * up_out  # [intermediate_size]

        # 路径2: SiLU值路径 = silu_gate * delta_up
        path2 = silu_gate * delta_up  # [intermediate_size]

        # 总中间层变化
        total_intermediate_delta = path1 + path2

        # 计算W_down对中间层变化的投影
        if hasattr(mlp, "down_proj"):
            W_down = mlp.down_proj.weight.detach().cpu().float()  # [hidden, intermediate]
        else:
            continue

        # W_down @ path1, W_down @ path2
        down_path1 = (W_down @ torch.tensor(path1, dtype=torch.float32)).numpy()
        down_path2 = (W_down @ torch.tensor(path2, dtype=torch.float32)).numpy()
        down_total = (W_down @ torch.tensor(total_intermediate_delta, dtype=torch.float32)).numpy()

        # 计算各路径的范数和在W_lm方向的投影
        r = {
            "path1_silu_grad_norm": float(np.linalg.norm(path1)),
            "path2_silu_val_norm": float(np.linalg.norm(path2)),
            "total_intermediate_delta_norm": float(np.linalg.norm(total_intermediate_delta)),
            "down_path1_norm": float(np.linalg.norm(down_path1)),
            "down_path2_norm": float(np.linalg.norm(down_path2)),
            "down_total_norm": float(np.linalg.norm(down_total)),
            "down_path1_proj_wlm": float(np.dot(down_path1, w_lm_norm_np)),
            "down_path2_proj_wlm": float(np.dot(down_path2, w_lm_norm_np)),
            "down_total_proj_wlm": float(np.dot(down_total, w_lm_norm_np)),
            "silu_grad_mean": float(np.mean(silu_jacobian)),
            "silu_val_mean": float(np.mean(np.abs(silu_gate))),
            "delta_gate_norm": float(np.linalg.norm(delta_gate)),
            "delta_up_norm": float(np.linalg.norm(delta_up)),
            "gate_out_norm": float(np.linalg.norm(gate_out)),
            "up_out_norm": float(np.linalg.norm(up_out)),
        }

        # 路径占比
        total_norm = r["down_path1_norm"] + r["down_path2_norm"]
        if total_norm > 1e-8:
            r["path1_frac"] = r["down_path1_norm"] / total_norm
            r["path2_frac"] = r["down_path2_norm"] / total_norm
        else:
            r["path1_frac"] = 0.0
            r["path2_frac"] = 0.0

        results[attr] = r
        print(f"  {attr}: path1(SiLU'_gate*Δgate*up)={r['down_path1_norm']:.4f} ({r['path1_frac']:.1%}), "
              f"path2(SiLU_gate*Δup)={r['down_path2_norm']:.4f} ({r['path2_frac']:.1%}), "
              f"silu'_mean={r['silu_grad_mean']:.3f}, silu_mean={r['silu_val_mean']:.3f}")

    return results


# ========== P379: 多层Jacobian全扫描 ==========

def run_p379(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P379: 多层Jacobian全扫描 - {model_name}")
    print(f"{'='*60}")

    results = {}
    all_layers = get_layers(model)  # 所有层
    n_layers = len(all_layers)
    print(f"  Total layers: {n_layers}")

    embed = model.get_input_embeddings()

    for attr in ATTRS[:1]:  # 只测1个属性词
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

        attr_results = []

        # 每隔几层采样(避免太慢)
        sample_layers = list(range(0, n_layers, max(1, n_layers // 15)))
        if n_layers - 1 not in sample_layers:
            sample_layers.append(n_layers - 1)

        for li in sample_layers:
            layer = all_layers[li]
            mlp = get_mlp(model, layer)
            mlp_dtype = next(mlp.parameters()).dtype

            # Hook: 获取MLP输入
            captured_mlp_base = {}
            captured_mlp_interv = {}
            captured_layer_base = {}
            captured_layer_interv = {}

            def make_mlp_base_hook(key):
                def hook(module, input, output):
                    if isinstance(input, tuple):
                        captured_mlp_base[key] = input[0].detach().float()
                    else:
                        captured_mlp_base[key] = input.detach().float()
                return hook

            def make_mlp_interv_hook(key):
                def hook(module, input, output):
                    if isinstance(input, tuple):
                        captured_mlp_interv[key] = input[0].detach().float()
                    else:
                        captured_mlp_interv[key] = input.detach().float()
                return hook

            def make_layer_base_hook(key):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        captured_layer_base[key] = output[0].detach().float()
                    else:
                        captured_layer_base[key] = output.detach().float()
                return hook

            def make_layer_interv_hook(key):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        captured_layer_interv[key] = output[0].detach().float()
                    else:
                        captured_layer_interv[key] = output.detach().float()
                return hook

            # 找layernorm
            if hasattr(layer, "post_attention_layernorm"):
                ln = layer.post_attention_layernorm
            else:
                ln = layer.input_layernorm

            # Base forward
            hooks_b = []
            hooks_b.append(ln.register_forward_hook(make_mlp_base_hook(f"L{li}")))
            hooks_b.append(layer.register_forward_hook(make_layer_base_hook(f"L{li}")))
            with torch.no_grad():
                try:
                    _ = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids)
                except:
                    for h in hooks_b: h.remove()
                    continue
            for h in hooks_b: h.remove()

            # Intervened forward
            hooks_i = []
            hooks_i.append(ln.register_forward_hook(make_mlp_interv_hook(f"L{li}")))
            hooks_i.append(layer.register_forward_hook(make_layer_interv_hook(f"L{li}")))
            with torch.no_grad():
                try:
                    _ = model(inputs_embeds=inputs_embeds_intervened, position_ids=position_ids)
                except:
                    for h in hooks_i: h.remove()
                    continue
            for h in hooks_i: h.remove()

            # 计算Jacobian范数
            J_W_norm = 0.0
            if f"L{li}" in captured_mlp_base:
                mlp_input = captured_mlp_base[f"L{li}"][0, -1, :]
                eps = 1e-3
                x_base = mlp_input.unsqueeze(0).unsqueeze(0).to(mlp_dtype)
                x_plus = x_base + eps * w_lm_tensor.to(mlp_dtype).unsqueeze(0).unsqueeze(0)

                with torch.no_grad():
                    try:
                        mlp_out_b = mlp(x_base)
                        if isinstance(mlp_out_b, tuple): mlp_out_b = mlp_out_b[0]
                        mlp_out_p = mlp(x_plus)
                        if isinstance(mlp_out_p, tuple): mlp_out_p = mlp_out_p[0]
                        J_W = (mlp_out_p - mlp_out_b) / eps
                        J_W_norm = float(J_W[0, 0, :].detach().float().cpu().norm().item())
                    except:
                        pass

            # 计算cos和proj
            cos_val = 0.0
            proj_val = 0.0
            delta_norm = 0.0
            if f"L{li}" in captured_layer_base and f"L{li}" in captured_layer_interv:
                h_b = captured_layer_base[f"L{li}"][0, -1, :].cpu().numpy()
                h_i = captured_layer_interv[f"L{li}"][0, -1, :].cpu().numpy()
                delta = h_i - h_b
                delta_norm = float(np.linalg.norm(delta))
                cos_val = float(np.dot(delta, w_lm_norm_np) / delta_norm) if delta_norm > 1e-8 else 0.0
                proj_val = float(np.dot(delta, w_lm_norm_np))

            lr = {
                "layer": li,
                "J_W_norm": J_W_norm,
                "cos_wlm": cos_val,
                "proj_wlm": proj_val,
                "delta_norm": delta_norm,
            }
            attr_results.append(lr)
            print(f"  {attr} L{li}/{n_layers-1}: ||J||={J_W_norm:.4f}, cos={cos_val:.4f}, "
                  f"proj={proj_val:.2f}, ||Δ||={delta_norm:.2f}")

        results[attr] = attr_results

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--experiment", type=str, default="all",
                       choices=["all", "p377", "p378", "p379"])
    args = parser.parse_args()

    model_name = args.model
    timestamp = time.strftime("%Y%m%d_%H%M")

    model, tokenizer, device = load_model(model_name)
    all_results = {"model": model_name, "timestamp": timestamp}

    try:
        if args.experiment in ["all", "p377"]:
            all_results["p377"] = run_p377(model, tokenizer, device, model_name)
        if args.experiment in ["all", "p378"]:
            all_results["p378"] = run_p378(model, tokenizer, device, model_name)
        if args.experiment in ["all", "p379"]:
            all_results["p379"] = run_p379(model, tokenizer, device, model_name)
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        out_file = OUT_DIR / f"phase_lxxiii_p377_379_{model_name}_{timestamp}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {out_file}")
        del model
        import gc; gc.collect(); torch.cuda.empty_cache()
        print("Model unloaded")


if __name__ == "__main__":
    main()
