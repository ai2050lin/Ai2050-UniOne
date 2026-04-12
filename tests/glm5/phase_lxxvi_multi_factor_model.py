"""
Phase LXXVI-P387/388/389: Multi-Factor Joint Model for Direction Preservation
======================================================================

Stage 2 of "Language Direction Preservation" complete theory:
  Stage 1 (done): W_up spectral norm -> Jacobian norm -> MLP interference -> direction preservation
  Stage 2 (this): Multi-factor joint model: W_gate, SiLU working point, W_down combined effects

P387: MLP Parameter Decomposition - Full Spectrum Analysis
  - Compute sigma(W_gate), sigma(W_up), sigma(W_down) for ALL layers L0-L(n_layers-1)
  - Compute gate_bias statistics for each layer
  - Compute the "effective gain" = sigma(W_down) * avg_silu_grad * sigma(W_up)
  - Find the key factor that determines cos decay rate

P388: Multi-Factor Regression - Building the Precise Formula
  - Target: cos(t) = cos(0) * exp(-lambda_eff * t)
  - lambda_eff = f(sigma(W_up), sigma(W_gate), sigma(W_down), gate_bias, avg_silu_grad)
  - Try different functional forms:
    a) lambda = c1 * sigma(W_up) / W_lm_norm   (current model)
    b) lambda = c1 * sigma(W_up) * sigma(W_gate) / (sigma(W_down) * W_lm_norm)
    c) lambda = c1 * sigma(W_up) / (sigma(W_down) * avg_silu_grad * W_lm_norm)
    d) lambda = c1 * effective_gain / W_lm_norm
  - Use least squares to fit c1 for each model, compare residuals

P389: Layer-by-Layer Prediction Validation
  - Use the best formula from P388 to predict cos at each layer
  - Compare predicted vs actual cos for all three models
  - Compute R^2 and mean absolute error
  - If R^2 > 0.95, the model is validated

Models: qwen3 -> glm4 -> deepseek7b (sequential)
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

PROMPT = "The apple is"
ATTRS = ["red", "blue", "big", "small", "hot", "cold"]

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


def get_layers(model):
    """Get model layers regardless of architecture."""
    if hasattr(model.model, "layers"):
        return list(model.model.layers)
    elif hasattr(model.model, "encoder"):
        return list(model.model.encoder.layers)
    else:
        raise ValueError("Cannot find layers")


def get_mlp_weights(model, model_name):
    """Extract W_gate, W_up, W_down, b_in from all MLP layers.
    
    Handles different architectures:
    - Standard (Qwen/DS): gate_proj, up_proj, down_proj (separate)
    - GLM4: gate_up_proj (merged), down_proj
    - TransformerLens: W_gate, W_in, W_out
    """
    weights = {}
    layers = get_layers(model)
    
    for i, layer in enumerate(layers):
        # Get MLP module
        if hasattr(layer, "mlp"):
            mlp = layer.mlp
        elif hasattr(layer, "feed_forward"):
            mlp = layer.feed_forward
        else:
            continue
        
        w_gate = None
        w_up = None
        w_down = None
        b_in = None
        act_fn_name = "unknown"
        architecture = "standard"
        
        # GLM4 style: gate_up_proj is merged
        if hasattr(mlp, 'gate_up_proj'):
            merged = mlp.gate_up_proj.weight.data.cpu().float()
            # Split along output dimension: first half = gate, second half = up
            mid = merged.shape[0] // 2
            w_gate = merged[:mid, :]
            w_up = merged[mid:, :]
            architecture = "merged_gate_up"
        # Standard separate projections
        elif hasattr(mlp, 'gate_proj'):
            w_gate = mlp.gate_proj.weight.data.cpu().float()
        # TransformerLens style
        elif hasattr(mlp, 'W_gate'):
            w_gate = mlp.W_gate.data.cpu().float()
        
        # up_proj
        if w_up is None:  # Not yet set by merged path
            if hasattr(mlp, 'up_proj'):
                w_up = mlp.up_proj.weight.data.cpu().float()
            elif hasattr(mlp, 'W_in'):
                w_up = mlp.W_in.data.cpu().float()
        
        # down_proj
        if hasattr(mlp, 'down_proj'):
            w_down = mlp.down_proj.weight.data.cpu().float()
        elif hasattr(mlp, 'W_out'):
            w_down = mlp.W_out.data.cpu().float()
        
        if hasattr(mlp, 'b_in'):
            b_in = mlp.b_in.data.cpu().float()
        
        if hasattr(mlp, 'act_fn'):
            act_fn_name = str(mlp.act_fn)
        
        weights[f"L{i}"] = {
            "W_gate": w_gate,
            "W_up": w_up,
            "W_down": w_down,
            "b_in": b_in,
            "act_fn": act_fn_name,
            "architecture": architecture,
        }
    
    return weights


def compute_spectral_stats(W, top_k=20):
    """Compute spectral norm and stats of a weight matrix.
    Uses power iteration for spectral norm (fast) and optional SVD for top-k.
    """
    try:
        # Fast spectral norm via power iteration (much faster than full SVD)
        v = torch.randn(W.shape[1], dtype=W.dtype, device=W.device)
        for _ in range(30):
            u = W @ v
            u_norm = u.norm()
            if u_norm > 0:
                u = u / u_norm
            v = W.T @ u
            v_norm = v.norm()
            if v_norm > 0:
                v = v / v_norm
        spectral = float((W @ v).norm())
        
        # Frobenius norm
        frobenius = float(torch.norm(W, p='fro'))
        
        return {
            "spectral": spectral,
            "frobenius": frobenius,
        }
    except Exception as e:
        return {"error": str(e)}


def compute_silu_grad_stats(model, model_name, tok, device, layer_indices):
    """Compute SiLU gradient statistics for each layer.
    
    For GLM4 (merged gate_up_proj), we hook on the MLP's gate_up_proj
    to capture the pre-activation gate values from the split output.
    For standard architecture, we hook on gate_proj input.
    """
    prompt = PROMPT
    inputs = tok(prompt, return_tensors="pt").to(device)
    
    stats = {}
    
    with torch.no_grad():
        all_layers = get_layers(model)
        for i in layer_indices:
                if i >= len(all_layers):
                    continue
                layer = all_layers[i]
                mlp = layer.mlp if hasattr(layer, 'mlp') else layer.feed_forward
                
                # Determine architecture and hook target
                is_merged = hasattr(mlp, 'gate_up_proj')
                is_standard = hasattr(mlp, 'gate_proj')
                
                if is_merged:
                    # GLM4 style: hook on gate_up_proj output, split to get gate
                    captured = {}
                    def hook_fn_merged(module, input, output, layer_idx=i):
                        # output is [batch, seq, 2*intermediate_size]
                        # Split: first half = gate, second half = up
                        gate, _ = output.chunk(2, dim=-1)
                        captured[f"pre_gate_L{layer_idx}"] = gate.detach()
                    
                    handle = mlp.gate_up_proj.register_forward_hook(hook_fn_merged)
                    try:
                        _ = model(**inputs)
                    except:
                        pass
                    handle.remove()
                    
                elif is_standard:
                    # Standard: hook on gate_proj to capture input
                    captured = {}
                    def hook_fn(module, input, output, layer_idx=i):
                        captured[f"pre_gate_L{layer_idx}"] = input[0].detach()
                    
                    handle = mlp.gate_proj.register_forward_hook(hook_fn)
                    try:
                        _ = model(**inputs)
                    except:
                        pass
                    handle.remove()
                else:
                    continue  # Skip unsupported architecture
                
                if f"pre_gate_L{i}" in captured:
                    pre_gate = captured[f"pre_gate_L{i}"][0, -1, :].float()
                    # SiLU'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
                    sigmoid_x = torch.sigmoid(pre_gate)
                    silu_grad = sigmoid_x * (1 + pre_gate * (1 - sigmoid_x))
                    stats[f"L{i}"] = {
                        "pre_gate_mean": float(pre_gate.mean()),
                        "pre_gate_std": float(pre_gate.std()),
                        "pre_gate_abs_mean": float(pre_gate.abs().mean()),
                        "silu_grad_mean": float(silu_grad.mean()),
                        "silu_grad_std": float(silu_grad.std()),
                        "silu_grad_median": float(silu_grad.median()),
                        "silu_grad_q10": float(silu_grad.quantile(0.1)),
                        "silu_grad_q90": float(silu_grad.quantile(0.9)),
                        "sigmoid_mean": float(sigmoid_x.mean()),
                    }
    
    return stats


def compute_cos_decay_data(model, model_name, tok, device):
    """Compute actual cos decay curve by injecting direction at L0 and measuring at each layer."""
    prompt = PROMPT
    inputs = tok(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    
    # Get W_lm (lm_head weight) - move to CPU to save GPU memory
    W_lm = None
    if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
        W_lm = model.lm_head.weight.data.cpu().float()
    elif hasattr(model, 'model') and hasattr(model.model, 'output') and hasattr(model.model.output, 'weight'):
        W_lm = model.model.output.weight.data.cpu().float()
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'output'):
        W_lm = model.transformer.output.weight.data.cpu().float()
    else:
        # Try to find output projection via get_output_embeddings
        out_emb = model.get_output_embeddings()
        if out_emb is not None and hasattr(out_emb, 'weight'):
            W_lm = out_emb.weight.data.cpu().float()
    
    if W_lm is None:
        raise ValueError("Cannot find lm_head/output weight in model")
    
    # Choose a direction in W_lm space
    attr = "red"
    attr_id = tok.convert_tokens_to_ids(attr)
    w_lm_attr = W_lm[attr_id].detach()
    w_lm_norm = w_lm_attr / w_lm_attr.norm()
    
    # Hook to capture hidden states at each layer
    layers = get_layers(model)
    n_layers = len(layers)
    
    captured_base = {}
    captured_int = {}
    
    def make_hook(captured_dict, prefix):
        def hook_fn(module, input, output, _prefix=prefix):
            if isinstance(output, tuple):
                captured_dict[_prefix] = output[0].detach()
            else:
                captured_dict[_prefix] = output.detach()
        return hook_fn
    
    # Register hooks for base run
    handles = []
    for i in range(min(n_layers, 40)):
        h = layers[i].register_forward_hook(
            make_hook(captured_base, f"L{i}")
        )
        handles.append(h)
    
    # Base run
    with torch.no_grad():
        _ = model(input_ids)
    
    # Remove hooks
    for h in handles:
        h.remove()
    
    # Intervention run: inject direction at L0 output
    handles = []
    beta = 5.0
    for i in range(min(n_layers, 40)):
        if i == 0:
            # Combined inject + capture hook for L0
            def inject_and_capture_hook(module, input, output, _i=i):
                if isinstance(output, tuple):
                    h = output[0].clone()
                else:
                    h = output.clone()
                # Inject
                h[:, -1, :] += beta * w_lm_norm.to(h.dtype).to(h.device)
                # Capture
                captured_int[f"L{_i}"] = h.detach()
                return (h,) + output[1:] if isinstance(output, tuple) else h
            h = layers[i].register_forward_hook(inject_and_capture_hook)
        else:
            h = layers[i].register_forward_hook(
                make_hook(captured_int, f"L{i}")
            )
        handles.append(h)
    
    with torch.no_grad():
        _ = model(input_ids)
    
    for h in handles:
        h.remove()
    
    # Compute cos at each layer
    cos_data = {}
    w_lm_np = w_lm_attr.cpu().numpy()
    
    for key in captured_base.keys():
        if key not in captured_int:
            continue
        h_base = captured_base[key][0, -1, :].float().cpu().numpy()
        h_int = captured_int[key][0, -1, :].float().cpu().numpy()
        delta = h_int - h_base
        delta_norm = np.linalg.norm(delta)
        if delta_norm > 1e-8:
            proj = np.dot(delta, w_lm_np)
            cos = proj / (delta_norm * np.linalg.norm(w_lm_np))
            cos_data[key] = {
                "cos_wlm": float(cos),
                "proj_wlm": float(proj),
                "delta_norm": float(delta_norm),
            }
    
    return cos_data, float(w_lm_attr.norm()), float(W_lm.norm() / W_lm.shape[0])


def run_p387(model_name):
    """P387: MLP Parameter Decomposition - Full Spectrum Analysis."""
    print(f"\n{'='*60}")
    print(f"P387: MLP Parameter Decomposition - {model_name}")
    print(f"{'='*60}")
    
    model, tok, device = load_model(model_name)
    
    # 1. Get MLP weights
    weights = get_mlp_weights(model, model_name)
    n_layers = len(weights)
    print(f"  Found {n_layers} layers")
    
    # 2. Compute spectral stats for all layers
    layer_stats = {}
    scan_layers = list(range(min(n_layers, 40)))
    
    for i in scan_layers:
        key = f"L{i}"
        if key not in weights:
            continue
        w = weights[key]
        
        stats = {"layer": i}
        
        if w["W_gate"] is not None:
            gate_stats = compute_spectral_stats(w["W_gate"])
            stats["W_gate_spectral"] = gate_stats.get("spectral", 0)
            stats["W_gate_frobenius"] = gate_stats.get("frobenius", 0)
        else:
            stats["W_gate_spectral"] = 0
            stats["W_gate_frobenius"] = 0
        
        if w["W_up"] is not None:
            up_stats = compute_spectral_stats(w["W_up"])
            stats["W_up_spectral"] = up_stats.get("spectral", 0)
            stats["W_up_frobenius"] = up_stats.get("frobenius", 0)
        else:
            stats["W_up_spectral"] = 0
            stats["W_up_frobenius"] = 0
        
        if w["W_down"] is not None:
            down_stats = compute_spectral_stats(w["W_down"])
            stats["W_down_spectral"] = down_stats.get("spectral", 0)
            stats["W_down_frobenius"] = down_stats.get("frobenius", 0)
        else:
            stats["W_down_spectral"] = 0
            stats["W_down_frobenius"] = 0
        
        if w["b_in"] is not None:
            stats["b_in_mean"] = float(w["b_in"].mean())
            stats["b_in_std"] = float(w["b_in"].std())
            stats["b_in_abs_mean"] = float(w["b_in"].abs().mean())
        else:
            stats["b_in_mean"] = 0
            stats["b_in_std"] = 0
            stats["b_in_abs_mean"] = 0
        
        # Compute derived metrics
        su = stats["W_up_spectral"]
        sg = stats["W_gate_spectral"]
        sd = stats["W_down_spectral"]
        
        # Ratio metrics
        stats["gate_up_ratio"] = sg / su if su > 1e-8 else 0
        stats["down_up_ratio"] = sd / su if su > 1e-8 else 0
        stats["gate_down_ratio"] = sg / sd if sd > 1e-8 else 0
        
        # Effective gain = sigma(W_down) * sigma(W_up) / sigma(W_gate) * avg_silu_grad
        # (higher gate -> more gating -> less signal passes through)
        stats["effective_gain_raw"] = sd * su / sg if sg > 1e-8 else 0
        
        # MLP Jacobian upper bound: sigma(W_down) * sigma(W_up)
        stats["jacobian_upper_bound"] = sd * su
        
        layer_stats[key] = stats
        
        if i < 10 or i % 5 == 0:
            print(f"  L{i}: sigma(W_gate)={sg:.2f}, sigma(W_up)={su:.2f}, sigma(W_down)={sd:.2f}, "
                  f"gate/up={stats['gate_up_ratio']:.1f}, down/up={stats['down_up_ratio']:.1f}")
    
    # 3. Get W_lm stats - move to CPU to save GPU memory
    W_lm = None
    if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
        W_lm = model.lm_head.weight.data.cpu().float()
    elif hasattr(model, 'model') and hasattr(model.model, 'output') and hasattr(model.model.output, 'weight'):
        W_lm = model.model.output.weight.data.cpu().float()
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'output'):
        W_lm = model.transformer.output.weight.data.cpu().float()
    else:
        out_emb = model.get_output_embeddings()
        if out_emb is not None and hasattr(out_emb, 'weight'):
            W_lm = out_emb.weight.data.cpu().float()
    
    if W_lm is None:
        raise ValueError("Cannot find lm_head/output weight in model")
    
    w_lm_stats = {
        "shape": list(W_lm.shape),
        "mean_norm": float(W_lm.norm(dim=1).mean()),
        "std_norm": float(W_lm.norm(dim=1).std()),
    }
    # Compute spectral norm on CPU to avoid OOM
    w_lm_cpu = W_lm.cpu()
    try:
        w_lm_stats["spectral"] = float(torch.linalg.svdvals(w_lm_cpu)[0])
    except Exception:
        # Fallback: use power iteration for spectral norm
        v = torch.randn(w_lm_cpu.shape[1], dtype=torch.float32)
        for _ in range(50):
            u = w_lm_cpu @ v
            u_norm = u.norm()
            if u_norm > 0:
                u = u / u_norm
            v = w_lm_cpu.T @ u
            v_norm = v.norm()
            if v_norm > 0:
                v = v / v_norm
        w_lm_stats["spectral"] = float((w_lm_cpu @ v).norm())
    del w_lm_cpu
    print(f"\n  W_lm: shape={W_lm.shape}, mean_norm={w_lm_stats['mean_norm']:.3f}, "
          f"spectral={w_lm_stats['spectral']:.1f}")
    
    # 4. Compute average stats across layers
    all_su = [layer_stats[f"L{i}"]["W_up_spectral"] for i in scan_layers if f"L{i}" in layer_stats]
    all_sg = [layer_stats[f"L{i}"]["W_gate_spectral"] for i in scan_layers if f"L{i}" in layer_stats]
    all_sd = [layer_stats[f"L{i}"]["W_down_spectral"] for i in scan_layers if f"L{i}" in layer_stats]
    
    summary = {
        "avg_W_up_spectral": float(np.mean(all_su)),
        "avg_W_gate_spectral": float(np.mean(all_sg)),
        "avg_W_down_spectral": float(np.mean(all_sd)),
        "avg_gate_up_ratio": float(np.mean([layer_stats[f"L{i}"]["gate_up_ratio"] for i in scan_layers if f"L{i}" in layer_stats])),
        "avg_down_up_ratio": float(np.mean([layer_stats[f"L{i}"]["down_up_ratio"] for i in scan_layers if f"L{i}" in layer_stats])),
        "d_model": W_lm.shape[1],
        "W_lm_mean_norm": w_lm_stats["mean_norm"],
        "W_lm_spectral": w_lm_stats["spectral"],
    }
    
    print(f"\n  Summary: avg_sigma(W_up)={summary['avg_W_up_spectral']:.2f}, "
          f"avg_sigma(W_gate)={summary['avg_W_gate_spectral']:.2f}, "
          f"avg_sigma(W_down)={summary['avg_W_down_spectral']:.2f}")
    print(f"  avg_gate/up={summary['avg_gate_up_ratio']:.1f}, "
          f"avg_down/up={summary['avg_down_up_ratio']:.1f}")
    
    # Return model info for reuse
    return {
        "layer_stats": layer_stats,
        "w_lm_stats": w_lm_stats,
        "summary": summary,
        "_model": model,
        "_tok": tok,
        "_device": device,
    }


def run_p388(model_name, p387_data):
    """P388: Multi-Factor Regression - Building the Precise Formula."""
    print(f"\n{'='*60}")
    print(f"P388: Multi-Factor Regression - {model_name}")
    print(f"{'='*60}")
    
    # Reuse model from P387 if available
    model = p387_data.pop("_model", None)
    tok = p387_data.pop("_tok", None)
    device = p387_data.pop("_device", None)
    
    if model is None:
        model, tok, device = load_model(model_name)
    
    # 1. Get actual cos decay data
    cos_data, w_lm_attr_norm, w_lm_mean_norm = compute_cos_decay_data(model, model_name, tok, device)
    
    print(f"\n  Actual cos decay:")
    actual_cos = {}
    for key in sorted(cos_data.keys(), key=lambda k: int(k[1:])):
        c = cos_data[key]["cos_wlm"]
        actual_cos[key] = c
        print(f"    {key}: cos={c:.4f}")
    
    # 2. Get SiLU gradient stats
    silu_stats = compute_silu_grad_stats(model, model_name, tok, device, 
                                          list(range(min(10, len(cos_data)))))
    print(f"\n  SiLU gradient stats:")
    for key in sorted(silu_stats.keys()):
        s = silu_stats[key]
        print(f"    {key}: silu_grad_mean={s['silu_grad_mean']:.4f}, std={s['silu_grad_std']:.4f}")
    
    # 3. Try different lambda formulas
    summary = p387_data["summary"]
    layer_stats = p387_data["layer_stats"]
    
    # Extract actual cos values as arrays
    layers_with_data = sorted([k for k in actual_cos.keys()], key=lambda k: int(k[1:]))
    cos_0 = actual_cos.get("L0", 0.5)
    
    # Compute cumulative sigma_up for each layer
    cum_sigma_up = {}
    cum_su = 0.0
    for key in layers_with_data:
        idx = int(key[1:])
        if key in layer_stats:
            cum_su += layer_stats[key]["W_up_spectral"]
        cum_sigma_up[key] = cum_su
    
    # Also compute cumulative sigma_gate, sigma_down, gate_up_ratio
    cum_sigma_gate = {}
    cum_sigma_down = {}
    cum_sg = 0.0
    cum_sd = 0.0
    for key in layers_with_data:
        if key in layer_stats:
            cum_sg += layer_stats[key]["W_gate_spectral"]
            cum_sd += layer_stats[key]["W_down_spectral"]
        cum_sigma_gate[key] = cum_sg
        cum_sigma_down[key] = cum_sd
    
    w_lm_norm = summary["W_lm_mean_norm"]
    avg_silu = 0.3  # default
    if "L0" in silu_stats:
        avg_silu = silu_stats["L0"]["silu_grad_mean"]
    elif silu_stats:
        first_key = list(silu_stats.keys())[0]
        avg_silu = silu_stats[first_key]["silu_grad_mean"]
    
    # Formula candidates:
    # a) cos(t) = cos(0) * exp(-c1 * cum_sigma_up / W_lm_norm)
    # b) cos(t) = cos(0) * exp(-c2 * cum_sigma_up * sigma_gate / (sigma_down * W_lm_norm))
    # c) cos(t) = cos(0) * exp(-c3 * cum_sigma_up / (sigma_down * silu_grad * W_lm_norm))
    # d) cos(t) = cos(0) * exp(-c4 * cum_effective_gain / W_lm_norm)
    # e) cos(t) = cos(0) * exp(-c5 * cum_sigma_up / (cum_sigma_down * W_lm_norm))
    # f) cos(t) = cos(0) * exp(-c6 * (cum_sigma_up / cum_sigma_gate) / W_lm_norm)
    
    # Fit each formula using least squares
    formulas = {}
    
    # Get data arrays for fitting (skip L0 since cos(0) is the starting point)
    fit_layers = [k for k in layers_with_data if k != "L0" and cum_sigma_up[k] > 0]
    if not fit_layers:
        fit_layers = layers_with_data[1:] if len(layers_with_data) > 1 else layers_with_data
    
    actual_cos_arr = np.array([actual_cos[k] for k in fit_layers])
    
    # Formula a: lambda = c1 * cum_sigma_up / W_lm_norm
    x_a = np.array([cum_sigma_up[k] / w_lm_norm for k in fit_layers])
    # cos = cos_0 * exp(-c1 * x_a)  =>  log(cos/cos_0) = -c1 * x_a
    log_ratio = np.log(np.maximum(actual_cos_arr / cos_0, 1e-10))
    valid = np.isfinite(log_ratio) & np.isfinite(x_a) & (x_a > 0)
    if valid.sum() > 0:
        c1, residuals_a = np.linalg.lstsq(x_a[valid].reshape(-1, 1), -log_ratio[valid], rcond=None)[:2]
        c1 = float(c1[0]) if len(c1) > 0 else 0.023
        predicted_a_all = cos_0 * np.exp(-c1 * np.array([cum_sigma_up[k] / w_lm_norm for k in layers_with_data]))
        # Compute R2 on fit_layers only (skip L0)
        pred_fit = cos_0 * np.exp(-c1 * x_a)
        ss_res = np.sum((actual_cos_arr - pred_fit)**2)
        ss_tot = np.sum((actual_cos_arr - actual_cos_arr.mean())**2)
        r2_a = 1 - ss_res / max(ss_tot, 1e-10)
    else:
        c1 = 0.023
        r2_a = 0.0
    
    formulas["a_cum_sigma_up_over_Wlm"] = {
        "formula": "cos(t) = cos(0) * exp(-c1 * cum_sigma_up / W_lm_norm)",
        "c1": c1,
        "R2": float(r2_a),
    }
    print(f"\n  Formula a: c1={c1:.4f}, R2={r2_a:.4f}")
    
    # Formula b: lambda = c2 * cum_sigma_up * avg_sigma_gate / (avg_sigma_down * W_lm_norm)
    avg_sg = summary["avg_W_gate_spectral"]
    avg_sd = summary["avg_W_down_spectral"]
    x_b = np.array([cum_sigma_up[k] * avg_sg / (avg_sd * w_lm_norm) for k in fit_layers])
    valid_b = np.isfinite(log_ratio) & np.isfinite(x_b) & (x_b > 0)
    if valid_b.sum() > 0:
        c2, _ = np.linalg.lstsq(x_b[valid_b].reshape(-1, 1), -log_ratio[valid_b], rcond=None)[:2]
        c2 = float(c2[0]) if len(c2) > 0 else 0.0
        pred_fit_b = cos_0 * np.exp(-c2 * x_b)
        ss_res_b = np.sum((actual_cos_arr - pred_fit_b)**2)
        ss_tot_b = np.sum((actual_cos_arr - actual_cos_arr.mean())**2)
        r2_b = 1 - ss_res_b / max(ss_tot_b, 1e-10)
    else:
        c2 = 0.0
        r2_b = 0.0
    
    formulas["b_sigma_up_gate_over_down_Wlm"] = {
        "formula": "cos(t) = cos(0) * exp(-c2 * cum_sigma_up * sigma_gate / (sigma_down * W_lm_norm))",
        "c2": c2,
        "R2": float(r2_b),
    }
    print(f"  Formula b: c2={c2:.4f}, R2={r2_b:.4f}")
    
    # Formula c: lambda = c3 * cum_sigma_up / (avg_sigma_down * avg_silu * W_lm_norm)
    x_c = np.array([cum_sigma_up[k] / (avg_sd * avg_silu * w_lm_norm) for k in fit_layers])
    valid_c = np.isfinite(log_ratio) & np.isfinite(x_c) & (x_c > 0)
    if valid_c.sum() > 0:
        c3, _ = np.linalg.lstsq(x_c[valid_c].reshape(-1, 1), -log_ratio[valid_c], rcond=None)[:2]
        c3 = float(c3[0]) if len(c3) > 0 else 0.0
        pred_fit_c = cos_0 * np.exp(-c3 * x_c)
        ss_res_c = np.sum((actual_cos_arr - pred_fit_c)**2)
        ss_tot_c = np.sum((actual_cos_arr - actual_cos_arr.mean())**2)
        r2_c = 1 - ss_res_c / max(ss_tot_c, 1e-10)
    else:
        c3 = 0.0
        r2_c = 0.0
    
    formulas["c_sigma_up_over_down_silu_Wlm"] = {
        "formula": "cos(t) = cos(0) * exp(-c3 * cum_sigma_up / (sigma_down * silu_grad * W_lm_norm))",
        "c3": c3,
        "R2": float(r2_c),
    }
    print(f"  Formula c: c3={c3:.4f}, R2={r2_c:.4f}")
    
    # Formula d: lambda = c4 * cum_effective_gain / W_lm_norm
    # effective_gain = sigma_down * sigma_up / sigma_gate
    cum_eff_gain = {}
    cum_eg = 0.0
    for key in layers_with_data:
        if key in layer_stats:
            su_l = layer_stats[key]["W_up_spectral"]
            sg_l = layer_stats[key]["W_gate_spectral"]
            sd_l = layer_stats[key]["W_down_spectral"]
            cum_eg += sd_l * su_l / sg_l if sg_l > 1e-8 else 0
        cum_eff_gain[key] = cum_eg
    
    x_d = np.array([cum_eff_gain[k] / w_lm_norm for k in fit_layers])
    valid_d = np.isfinite(log_ratio) & np.isfinite(x_d) & (x_d > 0)
    if valid_d.sum() > 0:
        c4, _ = np.linalg.lstsq(x_d[valid_d].reshape(-1, 1), -log_ratio[valid_d], rcond=None)[:2]
        c4 = float(c4[0]) if len(c4) > 0 else 0.0
        pred_fit_d = cos_0 * np.exp(-c4 * x_d)
        ss_res_d = np.sum((actual_cos_arr - pred_fit_d)**2)
        ss_tot_d = np.sum((actual_cos_arr - actual_cos_arr.mean())**2)
        r2_d = 1 - ss_res_d / max(ss_tot_d, 1e-10)
    else:
        c4 = 0.0
        r2_d = 0.0
    
    formulas["d_effective_gain_over_Wlm"] = {
        "formula": "cos(t) = cos(0) * exp(-c4 * cum(sigma_down*sigma_up/sigma_gate) / W_lm_norm)",
        "c4": c4,
        "R2": float(r2_d),
    }
    print(f"  Formula d: c4={c4:.4f}, R2={r2_d:.4f}")
    
    # Formula e: lambda = c5 * cum_sigma_up / (cum_sigma_down * W_lm_norm)
    x_e = np.array([cum_sigma_up[k] / (cum_sigma_down[k] * w_lm_norm) if cum_sigma_down[k] > 0 else 0 for k in fit_layers])
    valid_e = np.isfinite(log_ratio) & np.isfinite(x_e) & (x_e > 0)
    if valid_e.sum() > 0:
        c5, _ = np.linalg.lstsq(x_e[valid_e].reshape(-1, 1), -log_ratio[valid_e], rcond=None)[:2]
        c5 = float(c5[0]) if len(c5) > 0 else 0.0
        pred_fit_e = cos_0 * np.exp(-c5 * x_e)
        ss_res_e = np.sum((actual_cos_arr - pred_fit_e)**2)
        ss_tot_e = np.sum((actual_cos_arr - actual_cos_arr.mean())**2)
        r2_e = 1 - ss_res_e / max(ss_tot_e, 1e-10)
    else:
        c5 = 0.0
        r2_e = 0.0
    
    formulas["e_cum_sigma_up_over_cum_sigma_down_Wlm"] = {
        "formula": "cos(t) = cos(0) * exp(-c5 * cum_sigma_up / (cum_sigma_down * W_lm_norm))",
        "c5": c5,
        "R2": float(r2_e),
    }
    print(f"  Formula e: c5={c5:.4f}, R2={r2_e:.4f}")
    
    # Formula f: lambda = c6 * cum(sigma_up/sigma_gate) / W_lm_norm
    cum_su_sg = {}
    cum_val = 0.0
    for key in layers_with_data:
        if key in layer_stats:
            su_l = layer_stats[key]["W_up_spectral"]
            sg_l = layer_stats[key]["W_gate_spectral"]
            cum_val += su_l / sg_l if sg_l > 1e-8 else 0
        cum_su_sg[key] = cum_val
    
    x_f = np.array([cum_su_sg[k] / w_lm_norm for k in fit_layers])
    valid_f = np.isfinite(log_ratio) & np.isfinite(x_f) & (x_f > 0)
    if valid_f.sum() > 0:
        c6, _ = np.linalg.lstsq(x_f[valid_f].reshape(-1, 1), -log_ratio[valid_f], rcond=None)[:2]
        c6 = float(c6[0]) if len(c6) > 0 else 0.0
        pred_fit_f = cos_0 * np.exp(-c6 * x_f)
        ss_res_f = np.sum((actual_cos_arr - pred_fit_f)**2)
        ss_tot_f = np.sum((actual_cos_arr - actual_cos_arr.mean())**2)
        r2_f = 1 - ss_res_f / max(ss_tot_f, 1e-10)
    else:
        c6 = 0.0
        r2_f = 0.0
    
    formulas["f_cum_sigma_up_over_sigma_gate_Wlm"] = {
        "formula": "cos(t) = cos(0) * exp(-c6 * cum(sigma_up/sigma_gate) / W_lm_norm)",
        "c6": c6,
        "R2": float(r2_f),
    }
    print(f"  Formula f: c6={c6:.4f}, R2={r2_f:.4f}")
    
    # Find best formula
    best_formula = max(formulas.keys(), key=lambda k: formulas[k]["R2"])
    print(f"\n  *** BEST FORMULA: {best_formula} ***")
    print(f"  R2 = {formulas[best_formula]['R2']:.4f}")
    
    return {
        "cos_data": cos_data,
        "silu_stats": silu_stats,
        "cum_sigma_up": {k: float(v) for k, v in cum_sigma_up.items()},
        "cum_sigma_gate": {k: float(v) for k, v in cum_sigma_gate.items()},
        "cum_sigma_down": {k: float(v) for k, v in cum_sigma_down.items()},
        "formulas": formulas,
        "best_formula": best_formula,
        "_model": model,
        "_tok": tok,
        "_device": device,
    }


def run_p389(model_name, p387_data, p388_data):
    """P389: Layer-by-Layer Prediction Validation."""
    print(f"\n{'='*60}")
    print(f"P389: Layer-by-Layer Prediction Validation - {model_name}")
    print(f"{'='*60}")
    
    cos_data = p388_data["cos_data"]
    cum_sigma_up = p388_data["cum_sigma_up"]
    formulas = p388_data["formulas"]
    best_name = p388_data["best_formula"]
    best = formulas[best_name]
    layer_stats = p387_data["layer_stats"]
    w_lm_norm = p387_data["summary"]["W_lm_mean_norm"]
    
    # Get actual cos values
    layers_sorted = sorted(cos_data.keys(), key=lambda k: int(k[1:]))
    actual_cos = {k: cos_data[k]["cos_wlm"] for k in layers_sorted}
    cos_0 = actual_cos.get("L0", 0.5)
    
    print(f"\n  Best formula: {best['formula']}")
    print(f"  Coefficient: c = {best.get('c1', best.get('c2', best.get('c3', best.get('c4', best.get('c5', best.get('c6', 0))))))}")
    print(f"\n  Layer-by-layer comparison:")
    print(f"  {'Layer':<6} {'Actual cos':<12} {'Predicted cos':<14} {'Abs Error':<12} {'Rel Error'}")
    
    total_error = 0
    total_actual = 0
    n_points = 0
    
    for key in layers_sorted:
        idx = int(key[1:])
        actual = actual_cos[key]
        
        # Compute predicted cos using best formula
        # Extract coefficient from best dict
        c = best.get("c1", best.get("c2", best.get("c3", best.get("c4", best.get("c5", best.get("c6", 0.023))))))
        
        if best_name.startswith("a_"):
            x = cum_sigma_up.get(key, 0) / w_lm_norm
        elif best_name.startswith("b_"):
            avg_sg = p387_data["summary"]["avg_W_gate_spectral"]
            avg_sd = p387_data["summary"]["avg_W_down_spectral"]
            x = cum_sigma_up.get(key, 0) * avg_sg / (avg_sd * w_lm_norm)
        elif best_name.startswith("c_"):
            avg_sd = p387_data["summary"]["avg_W_down_spectral"]
            avg_silu = 0.3
            silu_stats = p388_data.get("silu_stats", {})
            if "L0" in silu_stats:
                avg_silu = silu_stats["L0"]["silu_grad_mean"]
            x = cum_sigma_up.get(key, 0) / (avg_sd * avg_silu * w_lm_norm)
        elif best_name.startswith("d_"):
            cum_eff = 0
            for k2 in layers_sorted:
                if int(k2[1:]) <= idx and k2 in layer_stats:
                    su = layer_stats[k2]["W_up_spectral"]
                    sg = layer_stats[k2]["W_gate_spectral"]
                    sd = layer_stats[k2]["W_down_spectral"]
                    cum_eff += sd * su / sg if sg > 1e-8 else 0
            x = cum_eff / w_lm_norm
        elif best_name.startswith("e_"):
            cum_sd = p388_data["cum_sigma_down"].get(key, 1)
            x = cum_sigma_up.get(key, 0) / (max(cum_sd, 1e-8) * w_lm_norm)
        elif best_name.startswith("f_"):
            cum_sg = 0
            for k2 in layers_sorted:
                if int(k2[1:]) <= idx and k2 in layer_stats:
                    su = layer_stats[k2]["W_up_spectral"]
                    sg = layer_stats[k2]["W_gate_spectral"]
                    cum_sg += su / sg if sg > 1e-8 else 0
            x = cum_sg / w_lm_norm
        else:
            x = cum_sigma_up.get(key, 0) / w_lm_norm
            c = best.get("c1", 0.023)
        
        predicted = cos_0 * np.exp(-c * x)
        abs_error = abs(predicted - actual)
        rel_error = abs_error / max(abs(actual), 1e-8)
        
        total_error += abs_error
        total_actual += abs(actual)
        n_points += 1
        
        print(f"  {key:<6} {actual:<12.4f} {predicted:<14.4f} {abs_error:<12.4f} {rel_error:.2%}")
    
    mae = total_error / max(n_points, 1)
    mre = total_error / max(total_actual, 1)
    
    print(f"\n  MAE (Mean Absolute Error) = {mae:.4f}")
    print(f"  MRE (Mean Relative Error) = {mre:.2%}")
    print(f"  Points = {n_points}")
    
    return {
        "mae": float(mae),
        "mre": float(mre),
        "n_points": n_points,
        "validation_ok": mre < 0.15,  # 15% relative error threshold
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=str, default="all", choices=["p387", "p388", "p389", "all"])
    args = parser.parse_args()
    
    model_name = args.model
    results = {"model": model_name}
    
    # Track model for cleanup
    model_ref = [None]
    
    try:
        if args.exp in ["p387", "all"]:
            p387_data = run_p387(model_name)
            results["p387"] = p387_data
            
            if args.exp == "all":
                p388_data = run_p388(model_name, p387_data)
                results["p388"] = p388_data
                
                p389_data = run_p389(model_name, p387_data, p388_data)
                results["p389"] = p389_data
                
                # Cleanup model from p388
                m = p388_data.pop("_model", None)
                if m is not None:
                    model_ref[0] = m
        elif args.exp == "p388":
            p387_data = run_p387(model_name)
            results["p387"] = p387_data
            p388_data = run_p388(model_name, p387_data)
            results["p388"] = p388_data
        elif args.exp == "p389":
            p387_data = run_p387(model_name)
            results["p387"] = p387_data
            p388_data = run_p388(model_name, p387_data)
            results["p388"] = p388_data
            p389_data = run_p389(model_name, p387_data, p388_data)
            results["p389"] = p389_data
    
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        results["error"] = str(e)
    
    # Cleanup
    if model_ref[0] is not None:
        del model_ref[0]
    torch.cuda.empty_cache()
    
    # Save results
    ts = time.strftime("%Y%m%d_%H%M", time.localtime())
    out_file = OUT_DIR / f"phase_lxxvi_p387_389_{model_name}_{ts}.json"
    
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items() if not k.startswith("_")}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif hasattr(obj, '__module__') and 'torch' in str(type(obj)):
            return f"<torch_object>"
        return obj
    
    results = convert(results)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
