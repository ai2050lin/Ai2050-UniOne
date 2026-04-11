"""
Phase LXXI-P371/372/373: MLP对注入方向的Jacobian敏感度分析
======================================================================

核心目标: 量化MLP对注入方向的"敏感度"，解释为什么GLM4的MLP输出Δ范数=0.30而Qwen3=10.51

P371: MLP的Jacobian向量分析
  - J_W = ∂MLP/∂x · W_lm[attr] (Jacobian在W_lm方向上的投影)
  - 如果||J_W||在GLM4中很小 → MLP对W_lm方向不敏感
  - 如果||J_W||在Qwen3中很大 → MLP对W_lm方向敏感
  - 同时计算Jacobian在多个方向的响应，看W_lm是否是"低敏感度方向"

P372: MLP各子层的敏感度分解
  - MLP(x) = W_down(act(W_gate(x)) * W_up(x))
  - 分解: ∂MLP/∂x = W_down · diag(act'(W_gate(x))*W_up(x) + act(W_gate(x))) · [W_gate; W_up]
  - 分别计算W_down、W_up、W_gate对W_lm方向的响应
  - 定位"哪个子层导致了GLM4的低敏感度"

P373: 残差连接的信号保真度定量分析
  - 对所有层L0-L5计算:
    - cos(Δh_Li, W_lm[attr]) (方向保持度)
    - ||ΔMLP_Li|| / ||Δh_residual|| (MLP干扰强度)
  - 验证: 干扰强度<10%的层是否cos>0.95?
  - 这将确认"MLP干扰强度"是跨层通用的预测指标

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

# 输出目录
OUT_DIR = Path("tests/glm5_temp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 属性词
ATTRS = ["red", "blue", "big", "small", "hot", "cold"]
PROMPT = "The apple is"

# 模型配置
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
        cfg["path"],
        dtype=torch.bfloat16,
        trust_remote_code=cfg["trust_remote_code"],
        local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        device_map="cpu",
    )
    if torch.cuda.is_available():
        mdl = mdl.to("cuda")
    mdl.eval()
    device = mdl.device

    tok = AutoTokenizer.from_pretrained(
        cfg["path"],
        trust_remote_code=cfg["trust_remote_code"],
        local_files_only=True,
        use_fast=cfg["use_fast"],
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    return mdl, tok, device


def get_w_lm_normed(model, tokenizer, attr):
    """获取W_lm[attr]并归一化"""
    attr_tok_ids = tokenizer.encode(attr, add_special_tokens=False)
    attr_tok_id = attr_tok_ids[0]
    lm_head = model.lm_head
    w_lm_attr = lm_head.weight[attr_tok_id].detach().cpu().float()
    w_lm_norm = w_lm_attr / w_lm_attr.norm()
    return w_lm_norm.numpy(), w_lm_attr.numpy()


def get_layer0(model):
    """获取Layer 0"""
    if hasattr(model.model, "layers"):
        return model.model.layers[0]
    elif hasattr(model.model, "encoder"):
        return model.model.encoder.layers[0]
    else:
        raise ValueError("Cannot find layers")


def get_layers(model, max_layers=6):
    """获取前N层"""
    if hasattr(model.model, "layers"):
        return list(model.model.layers[:max_layers])
    elif hasattr(model.model, "encoder"):
        return list(model.model.encoder.layers[:max_layers])
    else:
        raise ValueError("Cannot find layers")


def get_mlp(model, layer):
    """获取MLP模块"""
    if hasattr(layer, "mlp"):
        return layer.mlp
    elif hasattr(layer, "feed_forward"):
        return layer.feed_forward
    else:
        raise ValueError("Cannot find MLP")


# ========== P371: MLP Jacobian分析 ==========

def run_p371(model, tokenizer, device, model_name):
    """P371: MLP的Jacobian向量分析"""
    print(f"\n{'='*60}")
    print(f"P371: MLP Jacobian向量分析 - {model_name}")
    print(f"{'='*60}")

    results = {}
    layer0 = get_layer0(model)
    mlp = get_mlp(model, layer0)
    embed = model.get_input_embeddings()

    for attr in ATTRS:
        w_lm_norm_np, w_lm_np = get_w_lm_normed(model, tokenizer, attr)
        w_lm_tensor = torch.tensor(w_lm_norm_np, dtype=torch.float32, device=device)
        w_lm_raw_tensor = torch.tensor(w_lm_np, dtype=torch.float32, device=device)

        # 准备输入
        toks = tokenizer(PROMPT, return_tensors="pt").to(device)
        input_ids = toks.input_ids
        seq_len = input_ids.shape[1]
        inputs_embeds = embed(input_ids).detach().clone().to(model.dtype)

        # 获取L0的post_attention_layernorm输出作为MLP输入
        # 需要先跑一次base forward来获取MLP的输入
        captured_mlp_input = {}

        def capture_mlp_input_hook(module, input, output):
            if isinstance(input, tuple):
                captured_mlp_input["mlp_input"] = input[0].detach().float()
            else:
                captured_mlp_input["mlp_input"] = input.detach().float()

        # 找到post_attention_layernorm
        if hasattr(layer0, "post_attention_layernorm"):
            hook_target = layer0.post_attention_layernorm
        elif hasattr(layer0, "input_layernorm"):
            # 有些模型结构不同，用input_layernorm
            hook_target = layer0.input_layernorm
        else:
            print(f"  {attr}: Cannot find layernorm, skipping")
            continue

        h = hook_target.register_forward_hook(capture_mlp_input_hook)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        with torch.no_grad():
            _ = model(inputs_embeds=inputs_embeds, position_ids=position_ids)
        h.remove()

        if "mlp_input" not in captured_mlp_input:
            print(f"  {attr}: Failed to capture MLP input, skipping")
            continue

        mlp_input = captured_mlp_input["mlp_input"]  # [1, seq_len, hidden_dim] (float32)
        last_token_input = mlp_input[0, -1, :]  # [hidden_dim] (float32)

        # 计算MLP的Jacobian在W_lm方向上的投影
        # 用有限差分: J_W ≈ (MLP(x + eps*W_lm) - MLP(x)) / eps
        eps = 1e-3

        # 注意: MLP权重可能是bfloat16, 所以输入需要匹配dtype
        mlp_dtype = next(mlp.parameters()).dtype
        
        x_base = last_token_input.clone().unsqueeze(0).unsqueeze(0).to(mlp_dtype)  # [1, 1, hidden_dim]
        x_plus = x_base + eps * w_lm_tensor.unsqueeze(0).unsqueeze(0).to(mlp_dtype)

        with torch.no_grad():
            try:
                mlp_out_base = mlp(x_base)
                if isinstance(mlp_out_base, tuple):
                    mlp_out_base = mlp_out_base[0]
            except Exception as e:
                print(f"  {attr}: MLP forward failed: {e}")
                continue

            mlp_out_plus = mlp(x_plus)
            if isinstance(mlp_out_plus, tuple):
                mlp_out_plus = mlp_out_plus[0]

        J_W = (mlp_out_plus - mlp_out_base) / eps  # [1, 1, hidden_dim]
        J_W_vec = J_W[0, 0, :].detach().float().cpu().numpy()  # [hidden_dim]

        # 计算J_W的范数和在W_lm方向的投影
        J_W_norm = np.linalg.norm(J_W_vec)
        J_W_proj_wlm = np.dot(J_W_vec, w_lm_norm_np)
        J_W_cos_wlm = J_W_proj_wlm / J_W_norm if J_W_norm > 1e-8 else 0.0

        # 对比: Jacobian在随机方向上的响应
        np.random.seed(42)
        rand_dirs = []
        for _ in range(5):
            d = np.random.randn(len(w_lm_norm_np))
            d = d / np.linalg.norm(d)
            rand_dirs.append(d)

        rand_J_norms = []
        rand_J_projs = []
        for d in rand_dirs:
            d_tensor = torch.tensor(d, dtype=mlp_dtype, device=device)
            with torch.no_grad():
                x_plus_r = x_base + eps * d_tensor.unsqueeze(0).unsqueeze(0)
                mlp_out_plus_r = mlp(x_plus_r)
                if isinstance(mlp_out_plus_r, tuple):
                    mlp_out_plus_r = mlp_out_plus_r[0]
            J_r = (mlp_out_plus_r - mlp_out_base) / eps
            J_r_vec = J_r[0, 0, :].detach().float().cpu().numpy()
            rand_J_norms.append(np.linalg.norm(J_r_vec))
            rand_J_projs.append(np.dot(J_r_vec, w_lm_norm_np))

        result = {
            "attr": attr,
            "J_W_norm": float(J_W_norm),
            "J_W_proj_wlm": float(J_W_proj_wlm),
            "J_W_cos_wlm": float(J_W_cos_wlm),
            "rand_J_norm_mean": float(np.mean(rand_J_norms)),
            "rand_J_norm_std": float(np.std(rand_J_norms)),
            "rand_J_proj_wlm_mean": float(np.mean(rand_J_projs)),
            "w_lm_norm": float(np.linalg.norm(w_lm_np)),
        }

        # 判断W_lm是否是"低敏感度方向"
        z_score = (J_W_norm - np.mean(rand_J_norms)) / (np.std(rand_J_norms) + 1e-8)
        result["z_score_vs_rand"] = float(z_score)

        results[attr] = result
        print(f"  {attr}: ||J_W||={J_W_norm:.4f}, proj_wlm={J_W_proj_wlm:.4f}, "
              f"cos_wlm={J_W_cos_wlm:.4f}, rand_mean={np.mean(rand_J_norms):.4f}, "
              f"z_score={z_score:.2f}")

    return results


# ========== P372: MLP子层敏感度分解 ==========

def run_p372(model, tokenizer, device, model_name):
    """P372: MLP各子层对W_lm方向的敏感度分解"""
    print(f"\n{'='*60}")
    print(f"P372: MLP子层敏感度分解 - {model_name}")
    print(f"{'='*60}")

    results = {}
    layer0 = get_layer0(model)
    mlp = get_mlp(model, layer0)
    embed = model.get_input_embeddings()

    # 分析MLP结构
    print(f"  MLP type: {type(mlp)}")
    print(f"  MLP modules: {[n for n, _ in mlp.named_children()]}")

    for attr in ATTRS[:3]:  # 只测3个属性词以节省时间
        w_lm_norm_np, w_lm_np = get_w_lm_normed(model, tokenizer, attr)

        # 获取MLP输入
        toks = tokenizer(PROMPT, return_tensors="pt").to(device)
        input_ids = toks.input_ids
        seq_len = input_ids.shape[1]
        inputs_embeds = embed(input_ids).detach().clone().to(model.dtype)

        captured_mlp_input = {}

        def capture_mlp_input_hook(module, input, output):
            if isinstance(input, tuple):
                captured_mlp_input["mlp_input"] = input[0].detach().float()
            else:
                captured_mlp_input["mlp_input"] = input.detach().float()

        if hasattr(layer0, "post_attention_layernorm"):
            hook_target = layer0.post_attention_layernorm
        elif hasattr(layer0, "input_layernorm"):
            hook_target = layer0.input_layernorm
        else:
            continue

        h = hook_target.register_forward_hook(capture_mlp_input_hook)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        with torch.no_grad():
            _ = model(inputs_embeds=inputs_embeds, position_ids=position_ids)
        h.remove()

        if "mlp_input" not in captured_mlp_input:
            continue

        mlp_input = captured_mlp_input["mlp_input"][0, -1, :]  # [hidden_dim]
        mlp_input_np = mlp_input.detach().cpu().numpy()

        eps = 1e-3
        w_lm_tensor = torch.tensor(w_lm_norm_np, dtype=torch.float32, device=device)
        mlp_dtype = next(mlp.parameters()).dtype

        # 逐层分析
        # 标准 MLP: gate_proj, up_proj, down_proj
        # 或: W_gate, W_up, W_down
        layer_results = {}

        # 1. W_up的敏感度: x → W_up(x) 对W_lm方向的响应
        w_lm_cpu = torch.tensor(w_lm_norm_np, dtype=torch.float32)
        if hasattr(mlp, "up_proj"):
            W_up = mlp.up_proj.weight.detach().cpu().float()  # [intermediate, hidden]
            # W_up · W_lm
            W_up_wlm = W_up @ w_lm_cpu
            layer_results["W_up_wlm_norm"] = float(W_up_wlm.norm().item())
            layer_results["W_up_wlm_mean"] = float(W_up_wlm.mean().abs().item())

        # 2. W_gate的敏感度
        if hasattr(mlp, "gate_proj"):
            W_gate = mlp.gate_proj.weight.detach().cpu().float()
            W_gate_wlm = W_gate @ w_lm_cpu
            layer_results["W_gate_wlm_norm"] = float(W_gate_wlm.norm().item())
            layer_results["W_gate_wlm_mean"] = float(W_gate_wlm.mean().abs().item())

        # 3. W_down的敏感度
        if hasattr(mlp, "down_proj"):
            W_down = mlp.down_proj.weight.detach().cpu().float()  # [hidden, intermediate]
            # W_down · v 对W_lm方向的投影
            # 先用有限差分
            x_base_p372 = mlp_input.clone().unsqueeze(0).unsqueeze(0).to(mlp_dtype)

            # 计算MLP中间激活
            with torch.no_grad():
                if hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj"):
                    gate_out = mlp.gate_proj(x_base_p372)
                    up_out = mlp.up_proj(x_base_p372)
                    # SiLU gating
                    if hasattr(torch.nn.functional, "silu"):
                        act_gate = torch.nn.functional.silu(gate_out)
                    else:
                        act_gate = gate_out * torch.sigmoid(gate_out)
                    intermediate = act_gate * up_out  # [1, 1, intermediate_size]
                    intermediate_base = intermediate[0, 0, :].detach().float().cpu().numpy()

                    # 下一步: down_proj(intermediate)
                    # Δintermediate / Δx · W_lm
                    x_plus = x_base_p372 + eps * w_lm_tensor.to(mlp_dtype).unsqueeze(0).unsqueeze(0)
                    gate_out_plus = mlp.gate_proj(x_plus)
                    up_out_plus = mlp.up_proj(x_plus)
                    act_gate_plus = torch.nn.functional.silu(gate_out_plus)
                    intermediate_plus = act_gate_plus * up_out_plus
                    intermediate_delta = (intermediate_plus - intermediate)[0, 0, :].detach().float().cpu().numpy()

                    # Δintermediate 在W_down奇异向量上的分布
                    # intermediate_delta的范数
                    layer_results["intermediate_delta_norm"] = float(np.linalg.norm(intermediate_delta))
                    # intermediate_delta中有多少分量被W_down投影到W_lm方向?
                    # W_down · intermediate_delta → [hidden], 然后cos with W_lm
                    W_down_delta = W_down @ torch.tensor(intermediate_delta, dtype=torch.float32)
                    W_down_delta_np = W_down_delta.numpy()
                    W_down_delta_cos = np.dot(W_down_delta_np, w_lm_norm_np) / (np.linalg.norm(W_down_delta_np) + 1e-8)
                    layer_results["W_down_delta_cos_wlm"] = float(W_down_delta_cos)
                    layer_results["W_down_delta_norm"] = float(np.linalg.norm(W_down_delta_np))

                    # 3a. gate和up各自的贡献
                    # gate的导数: SiLU'(z) = SiLU(z) + sigmoid(z)(1 - SiLU(z)/z)
                    # 简化: 用有限差分
                    gate_delta = (gate_out_plus - gate_out)[0, 0, :].detach().float().cpu().numpy()
                    up_delta = (up_out_plus - up_out)[0, 0, :].detach().float().cpu().numpy()
                    layer_results["gate_delta_norm"] = float(np.linalg.norm(gate_delta))
                    layer_results["up_delta_norm"] = float(np.linalg.norm(up_delta))

                    # gate_base和up_base的范数
                    layer_results["gate_base_norm"] = float(np.linalg.norm(gate_out[0, 0, :].detach().float().cpu().numpy()))
                    layer_results["up_base_norm"] = float(np.linalg.norm(up_out[0, 0, :].detach().float().cpu().numpy()))
                    layer_results["intermediate_base_norm"] = float(np.linalg.norm(intermediate_base))

        # 4. 如果MLP结构不同 (dense_h_to_4h / dense_4h_to_h)
        if hasattr(mlp, "dense_h_to_4h"):
            W_up_full = mlp.dense_h_to_4h.weight.detach().cpu().float()
            W_up_wlm = W_up_full @ w_lm_cpu
            layer_results["W_h4h_wlm_norm"] = float(W_up_wlm.norm().item())

        if hasattr(mlp, "dense_4h_to_h"):
            W_down_full = mlp.dense_4h_to_h.weight.detach().cpu().float()
            # 类似分析
            layer_results["W_4hh_shape"] = list(W_down_full.shape)

        results[attr] = layer_results
        print(f"  {attr}: {layer_results}")

    return results


# ========== P373: 跨层信号保真度分析 ==========

def run_p373(model, tokenizer, device, model_name):
    """P373: 跨层信号保真度分析 - L0到L5"""
    print(f"\n{'='*60}")
    print(f"P373: 跨层信号保真度分析 - {model_name}")
    print(f"{'='*60}")

    results = {}
    embed = model.get_input_embeddings()
    layers = get_layers(model, max_layers=6)
    n_layers = len(layers)

    for attr in ATTRS[:3]:
        w_lm_norm_np, w_lm_np = get_w_lm_normed(model, tokenizer, attr)
        w_lm_tensor = torch.tensor(w_lm_norm_np, dtype=torch.float32, device=device)

        toks = tokenizer(PROMPT, return_tensors="pt").to(device)
        input_ids = toks.input_ids
        seq_len = input_ids.shape[1]
        inputs_embeds_base = embed(input_ids).detach().clone().to(model.dtype)

        beta = 8.0
        inputs_embeds_intervened = inputs_embeds_base.clone()
        inputs_embeds_intervened[0, -1, :] += (beta * w_lm_tensor).to(model.dtype)

        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        layer_results = []

        for li in range(n_layers):
            layer = layers[li]

            # Hook每层的MLP和整层输出
            captured = {}

            def make_hook(key):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        captured[key] = output[0].detach().float()
                    else:
                        captured[key] = output.detach().float()
                return hook

            mlp = get_mlp(model, layer)

            hooks = []
            hooks.append(mlp.register_forward_hook(make_hook(f"mlp_L{li}")))
            hooks.append(layer.register_forward_hook(make_hook(f"layer_L{li}")))

            # Base forward
            captured_base = {}
            def make_base_hook(key):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        captured_base[key] = output[0].detach().float()
                    else:
                        captured_base[key] = output.detach().float()
                return hook

            hooks_base = []
            hooks_base.append(mlp.register_forward_hook(make_base_hook(f"mlp_L{li}")))
            hooks_base.append(layer.register_forward_hook(make_base_hook(f"layer_L{li}")))

            with torch.no_grad():
                try:
                    _ = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids)
                except:
                    for h in hooks_base: h.remove()
                    continue
            for h in hooks_base: h.remove()

            # Intervened forward
            captured_interv = {}
            def make_interv_hook(key):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        captured_interv[key] = output[0].detach().float()
                    else:
                        captured_interv[key] = output.detach().float()
                return hook

            hooks_interv = []
            hooks_interv.append(mlp.register_forward_hook(make_interv_hook(f"mlp_L{li}")))
            hooks_interv.append(layer.register_forward_hook(make_interv_hook(f"layer_L{li}")))

            with torch.no_grad():
                try:
                    _ = model(inputs_embeds=inputs_embeds_intervened, position_ids=position_ids)
                except:
                    for h in hooks_interv: h.remove()
                    continue
            for h in hooks_interv: h.remove()

            base_key = f"layer_L{li}"
            mlp_key = f"mlp_L{li}"

            if base_key not in captured_base or base_key not in captured_interv:
                continue

            # 层输出分析
            h_base = captured_base[base_key][0, -1, :].detach().float().cpu().numpy()
            h_interv = captured_interv[base_key][0, -1, :].detach().float().cpu().numpy()
            delta_layer = h_interv - h_base
            delta_layer_norm = np.linalg.norm(delta_layer)
            cos_layer = float(np.dot(delta_layer, w_lm_norm_np) / delta_layer_norm) if delta_layer_norm > 1e-8 else 0.0
            proj_layer = float(np.dot(delta_layer, w_lm_norm_np))

            # MLP输出分析
            mlp_delta_norm = 0.0
            if mlp_key in captured_base and mlp_key in captured_interv:
                mlp_base = captured_base[mlp_key][0, -1, :].detach().float().cpu().numpy()
                mlp_interv = captured_interv[mlp_key][0, -1, :].detach().float().cpu().numpy()
                delta_mlp = mlp_interv - mlp_base
                mlp_delta_norm = float(np.linalg.norm(delta_mlp))

            # 残差信号 ≈ delta_layer - delta_mlp - delta_attn
            # 简化: 残差投影 ≈ proj_layer (因为残差是主要分量)
            interference = mlp_delta_norm / (proj_layer + 1e-8) if proj_layer > 0 else float('inf')

            lr = {
                "layer": li,
                "cos_wlm": cos_layer,
                "delta_norm": float(delta_layer_norm),
                "proj_wlm": proj_layer,
                "mlp_delta_norm": mlp_delta_norm,
                "mlp_interference": interference,
            }
            layer_results.append(lr)
            print(f"  {attr} L{li}: cos={cos_layer:.4f}, ||Δ||={delta_layer_norm:.2f}, "
                  f"proj={proj_layer:.2f}, ||ΔMLP||={mlp_delta_norm:.2f}, "
                  f"interf={interference:.2%}")

        results[attr] = layer_results

    return results


# ========== 主函数 ==========

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--experiment", type=str, default="all",
                       choices=["all", "p371", "p372", "p373"])
    args = parser.parse_args()

    model_name = args.model
    timestamp = time.strftime("%Y%m%d_%H%M")

    # 加载模型
    model, tokenizer, device = load_model(model_name)

    all_results = {"model": model_name, "timestamp": timestamp}

    try:
        if args.experiment in ["all", "p371"]:
            r371 = run_p371(model, tokenizer, device, model_name)
            all_results["p371"] = r371

        if args.experiment in ["all", "p372"]:
            r372 = run_p372(model, tokenizer, device, model_name)
            all_results["p372"] = r372

        if args.experiment in ["all", "p373"]:
            r373 = run_p373(model, tokenizer, device, model_name)
            all_results["p373"] = r373

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

    finally:
        # 保存结果
        out_file = OUT_DIR / f"phase_lxxi_p371_373_{model_name}_{timestamp}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {out_file}")

        # 清理
        del model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print("Model unloaded, GPU memory cleared")


if __name__ == "__main__":
    main()
