"""Phase LXXVI - GLM4 and DeepSeek7B Multi-Factor Model (Memory Optimized)

Memory optimization:
- Process one experiment at a time, delete model between experiments
- Use CPU offloading for weight computations
- Limit scan layers to reduce memory
- Clear cache aggressively between phases
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

OUT_DIR = Path("tests/glm5_temp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPT = "The apple is"

MODEL_CONFIGS = {
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

MAX_LAYERS = 40  # Limit scan layers


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


def free_model(model):
    """Aggressively free model memory."""
    del model
    torch.cuda.empty_cache()
    import gc
    gc.collect()


def get_layers(model):
    if hasattr(model.model, "layers"):
        return list(model.model.layers)
    elif hasattr(model.model, "encoder"):
        return list(model.model.encoder.layers)
    else:
        raise ValueError("Cannot find layers")


def get_mlp_weights(model, model_name):
    """Extract W_gate, W_up, W_down from all MLP layers."""
    weights = {}
    layers = get_layers(model)
    
    for i, layer in enumerate(layers):
        if i >= MAX_LAYERS:
            break
            
        if hasattr(layer, "mlp"):
            mlp = layer.mlp
        elif hasattr(layer, "feed_forward"):
            mlp = layer.feed_forward
        else:
            continue
        
        w_gate = None
        w_up = None
        w_down = None
        architecture = "standard"
        
        if hasattr(mlp, 'gate_up_proj'):
            merged = mlp.gate_up_proj.weight.data.cpu().float()
            mid = merged.shape[0] // 2
            w_gate = merged[:mid, :]
            w_up = merged[mid:, :]
            architecture = "merged_gate_up"
        elif hasattr(mlp, 'gate_proj'):
            w_gate = mlp.gate_proj.weight.data.cpu().float()
        
        if w_up is None:
            if hasattr(mlp, 'up_proj'):
                w_up = mlp.up_proj.weight.data.cpu().float()
        
        if hasattr(mlp, 'down_proj'):
            w_down = mlp.down_proj.weight.data.cpu().float()
        
        weights[f"L{i}"] = {
            "W_gate": w_gate,
            "W_up": w_up,
            "W_down": w_down,
            "architecture": architecture,
        }
    
    return weights


def compute_spectral_stats(W):
    """Power iteration for spectral norm."""
    try:
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
        frobenius = float(torch.norm(W, p='fro'))
        return {"spectral": spectral, "frobenius": frobenius}
    except Exception as e:
        return {"error": str(e)}


def get_w_lm(model):
    """Get W_lm (lm_head weight), CPU float."""
    W_lm = None
    if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
        W_lm = model.lm_head.weight.data.cpu().float()
    elif hasattr(model, 'model') and hasattr(model.model, 'output') and hasattr(model.model.output, 'weight'):
        W_lm = model.model.output.weight.data.cpu().float()
    else:
        out_emb = model.get_output_embeddings()
        if out_emb is not None and hasattr(out_emb, 'weight'):
            W_lm = out_emb.weight.data.cpu().float()
    if W_lm is None:
        raise ValueError("Cannot find lm_head/output weight")
    return W_lm


def run_p387(model_name):
    """P387: MLP Parameter Decomposition."""
    print(f"\n{'='*60}")
    print(f"P387: MLP Parameter Decomposition - {model_name}")
    print(f"{'='*60}")
    
    model, tok, device = load_model(model_name)
    weights = get_mlp_weights(model, model_name)
    n_layers = len(weights)
    print(f"  Found {n_layers} layers")
    
    layer_stats = {}
    for i in range(min(n_layers, MAX_LAYERS)):
        key = f"L{i}"
        if key not in weights:
            continue
        w = weights[key]
        stats = {"layer": i}
        
        if w["W_gate"] is not None:
            gs = compute_spectral_stats(w["W_gate"])
            stats["W_gate_spectral"] = gs.get("spectral", 0)
            stats["W_gate_frobenius"] = gs.get("frobenius", 0)
        else:
            stats["W_gate_spectral"] = 0
            stats["W_gate_frobenius"] = 0
        
        if w["W_up"] is not None:
            us = compute_spectral_stats(w["W_up"])
            stats["W_up_spectral"] = us.get("spectral", 0)
            stats["W_up_frobenius"] = us.get("frobenius", 0)
        else:
            stats["W_up_spectral"] = 0
            stats["W_up_frobenius"] = 0
        
        if w["W_down"] is not None:
            ds = compute_spectral_stats(w["W_down"])
            stats["W_down_spectral"] = ds.get("spectral", 0)
            stats["W_down_frobenius"] = ds.get("frobenius", 0)
        else:
            stats["W_down_spectral"] = 0
            stats["W_down_frobenius"] = 0
        
        su = stats["W_up_spectral"]
        sg = stats["W_gate_spectral"]
        sd = stats["W_down_spectral"]
        stats["gate_up_ratio"] = sg / su if su > 1e-8 else 0
        stats["down_up_ratio"] = sd / su if su > 1e-8 else 0
        stats["effective_gain_raw"] = sd * su / sg if sg > 1e-8 else 0
        stats["jacobian_upper_bound"] = sd * su
        stats["architecture"] = w.get("architecture", "standard")
        
        layer_stats[key] = stats
        if i < 10 or i % 5 == 0:
            print(f"  L{i}: sg={sg:.2f}, su={su:.2f}, sd={sd:.2f}, g/u={stats['gate_up_ratio']:.1f}")
    
    # Free weight data
    del weights
    
    # W_lm stats
    W_lm = get_w_lm(model)
    w_lm_stats = {
        "shape": list(W_lm.shape),
        "mean_norm": float(W_lm.norm(dim=1).mean()),
        "std_norm": float(W_lm.norm(dim=1).std()),
    }
    # Power iteration for spectral norm
    v = torch.randn(W_lm.shape[1], dtype=torch.float32)
    for _ in range(50):
        u = W_lm @ v
        u_norm = u.norm()
        if u_norm > 0:
            u = u / u_norm
        v = W_lm.T @ u
        v_norm = v.norm()
        if v_norm > 0:
            v = v / v_norm
    w_lm_stats["spectral"] = float((W_lm @ v).norm())
    del W_lm
    
    all_su = [layer_stats[f"L{i}"]["W_up_spectral"] for i in range(min(n_layers, MAX_LAYERS)) if f"L{i}" in layer_stats]
    all_sg = [layer_stats[f"L{i}"]["W_gate_spectral"] for i in range(min(n_layers, MAX_LAYERS)) if f"L{i}" in layer_stats]
    all_sd = [layer_stats[f"L{i}"]["W_down_spectral"] for i in range(min(n_layers, MAX_LAYERS)) if f"L{i}" in layer_stats]
    
    summary = {
        "avg_W_up_spectral": float(np.mean(all_su)),
        "avg_W_gate_spectral": float(np.mean(all_sg)),
        "avg_W_down_spectral": float(np.mean(all_sd)),
        "avg_gate_up_ratio": float(np.mean([layer_stats[f"L{i}"]["gate_up_ratio"] for i in range(min(n_layers, MAX_LAYERS)) if f"L{i}" in layer_stats])),
        "avg_down_up_ratio": float(np.mean([layer_stats[f"L{i}"]["down_up_ratio"] for i in range(min(n_layers, MAX_LAYERS)) if f"L{i}" in layer_stats])),
        "W_lm_mean_norm": w_lm_stats["mean_norm"],
        "W_lm_spectral": w_lm_stats["spectral"],
    }
    
    print(f"\n  Summary: avg_su={summary['avg_W_up_spectral']:.2f}, "
          f"avg_sg={summary['avg_W_gate_spectral']:.2f}, avg_sd={summary['avg_W_down_spectral']:.2f}")
    
    # Save P387 separately and free model
    p387_result = {
        "layer_stats": layer_stats,
        "w_lm_stats": w_lm_stats,
        "summary": summary,
    }
    
    ts = time.strftime("%Y%m%d_%H%M", time.localtime())
    out_file = OUT_DIR / f"phase_lxxvi_p387_{model_name}_{ts}.json"
    save_json(p387_result, out_file)
    
    # Keep model for P388
    return p387_result, model, tok, device


def run_p388(model_name, model, tok, device, p387_data):
    """P388: Multi-Factor Regression."""
    print(f"\n{'='*60}")
    print(f"P388: Multi-Factor Regression - {model_name}")
    print(f"{'='*60}")
    
    # 1. Compute cos decay
    cos_data, w_lm_attr_norm, w_lm_mean_norm = compute_cos_decay(model, tok, device)
    
    print(f"\n  Actual cos decay:")
    actual_cos = {}
    for key in sorted(cos_data.keys(), key=lambda k: int(k[1:])):
        c = cos_data[key]["cos_wlm"]
        actual_cos[key] = c
        if int(key[1:]) < 10 or int(key[1:]) % 5 == 0:
            print(f"    {key}: cos={c:.4f}")
    
    # 2. SiLU gradient stats
    silu_stats = compute_silu_grad(model, tok, device, list(range(min(10, len(cos_data)))))
    
    # 3. Try different lambda formulas
    summary = p387_data["summary"]
    layer_stats = p387_data["layer_stats"]
    
    layers_with_data = sorted(actual_cos.keys(), key=lambda k: int(k[1:]))
    cos_0 = actual_cos.get("L0", 0.5)
    
    # Cumulative sigma
    cum_sigma_up = {}
    cum_sigma_gate = {}
    cum_sigma_down = {}
    cum_su = cum_sg = cum_sd = 0.0
    for key in layers_with_data:
        if key in layer_stats:
            cum_su += layer_stats[key]["W_up_spectral"]
            cum_sg += layer_stats[key]["W_gate_spectral"]
            cum_sd += layer_stats[key]["W_down_spectral"]
        cum_sigma_up[key] = cum_su
        cum_sigma_gate[key] = cum_sg
        cum_sigma_down[key] = cum_sd
    
    w_lm_norm = summary["W_lm_mean_norm"]
    avg_silu = 0.3
    if "L0" in silu_stats:
        avg_silu = silu_stats["L0"]["silu_grad_mean"]
    
    # Fit layers (skip L0)
    fit_layers = [k for k in layers_with_data if k != "L0" and cum_sigma_up[k] > 0]
    if not fit_layers:
        fit_layers = layers_with_data[1:] if len(layers_with_data) > 1 else layers_with_data
    
    actual_cos_arr = np.array([actual_cos[k] for k in fit_layers])
    log_ratio = np.log(np.maximum(actual_cos_arr / cos_0, 1e-10))
    
    formulas = {}
    
    # Formula a: lambda = c1 * cum_sigma_up / W_lm_norm
    formulas["a_cum_sigma_up_over_WlmNorm"] = fit_formula(
        "a", fit_layers, cum_sigma_up, w_lm_norm, log_ratio, actual_cos_arr, cos_0,
        "cos(t) = cos(0) * exp(-c1 * cum_sigma_up / W_lm_norm)")
    
    # Formula b: lambda = c2 * cum_sigma_up * sigma_gate / (sigma_down * W_lm_norm)
    avg_sg = summary["avg_W_gate_spectral"]
    avg_sd = summary["avg_W_down_spectral"]
    x_b = {k: cum_sigma_up[k] * avg_sg / (avg_sd * w_lm_norm) for k in fit_layers}
    formulas["b_sigma_up_gate_over_down_Wlm"] = fit_formula_custom(
        "b", fit_layers, x_b, log_ratio, actual_cos_arr, cos_0,
        "cos(t) = cos(0) * exp(-c2 * cum_su*sg/(sd*Wlm))")
    
    # Formula c: lambda = c3 * cum_sigma_up / (sigma_down * silu * W_lm_norm)
    x_c = {k: cum_sigma_up[k] / (avg_sd * avg_silu * w_lm_norm) for k in fit_layers}
    formulas["c_sigma_up_over_down_silu_Wlm"] = fit_formula_custom(
        "c", fit_layers, x_c, log_ratio, actual_cos_arr, cos_0,
        "cos(t) = cos(0) * exp(-c3 * cum_su/(sd*silu*Wlm))")
    
    # Formula d: cum_effective_gain / W_lm_norm
    cum_eff_gain = {}
    cum_eg = 0.0
    for key in layers_with_data:
        if key in layer_stats:
            su_l = layer_stats[key]["W_up_spectral"]
            sg_l = layer_stats[key]["W_gate_spectral"]
            sd_l = layer_stats[key]["W_down_spectral"]
            cum_eg += sd_l * su_l / sg_l if sg_l > 1e-8 else 0
        cum_eff_gain[key] = cum_eg
    
    x_d = {k: cum_eff_gain[k] / w_lm_norm for k in fit_layers}
    formulas["d_effective_gain_over_WlmNorm"] = fit_formula_custom(
        "d", fit_layers, x_d, log_ratio, actual_cos_arr, cos_0,
        "cos(t) = cos(0) * exp(-c4 * cum(sd*su/sg)/Wlm)")
    
    # Formula e: cum_sigma_up / (cum_sigma_down * W_lm_norm)
    x_e = {k: cum_sigma_up[k] / (max(cum_sigma_down[k], 1e-8) * w_lm_norm) for k in fit_layers}
    formulas["e_cum_su_over_cum_sd_Wlm"] = fit_formula_custom(
        "e", fit_layers, x_e, log_ratio, actual_cos_arr, cos_0,
        "cos(t) = cos(0) * exp(-c5 * cum_su/(cum_sd*Wlm))")
    
    best_formula = max(formulas.keys(), key=lambda k: formulas[k]["R2"])
    print(f"\n  *** BEST: {best_formula}, R2={formulas[best_formula]['R2']:.4f} ***")
    
    p388_result = {
        "cos_data": cos_data,
        "silu_stats": silu_stats,
        "cum_sigma_up": {k: float(v) for k, v in cum_sigma_up.items()},
        "cum_sigma_gate": {k: float(v) for k, v in cum_sigma_gate.items()},
        "cum_sigma_down": {k: float(v) for k, v in cum_sigma_down.items()},
        "formulas": formulas,
        "best_formula": best_formula,
    }
    
    ts = time.strftime("%Y%m%d_%H%M", time.localtime())
    out_file = OUT_DIR / f"phase_lxxvi_p388_{model_name}_{ts}.json"
    save_json(p388_result, out_file)
    
    return p388_result, model, tok, device


def run_p389(model_name, model, tok, device, p387_data, p388_data):
    """P389: Layer-by-Layer Prediction Validation."""
    print(f"\n{'='*60}")
    print(f"P389: Validation - {model_name}")
    print(f"{'='*60}")
    
    cos_data = p388_data["cos_data"]
    cum_sigma_up = p388_data["cum_sigma_up"]
    formulas = p388_data["formulas"]
    best_name = p388_data["best_formula"]
    best = formulas[best_name]
    layer_stats = p387_data["layer_stats"]
    w_lm_norm = p387_data["summary"]["W_lm_mean_norm"]
    
    layers_sorted = sorted(cos_data.keys(), key=lambda k: int(k[1:]))
    actual_cos = {k: cos_data[k]["cos_wlm"] for k in layers_sorted}
    cos_0 = actual_cos.get("L0", 0.5)
    
    print(f"  Best: {best['formula']}")
    
    c = best.get("c1", best.get("c2", best.get("c3", best.get("c4", best.get("c5", 0.023)))))
    
    total_error = 0
    total_actual = 0
    n_points = 0
    
    predictions = {}
    
    for key in layers_sorted:
        idx = int(key[1:])
        actual = actual_cos[key]
        
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
        else:
            x = cum_sigma_up.get(key, 0) / w_lm_norm
        
        predicted = cos_0 * np.exp(-c * x)
        abs_error = abs(predicted - actual)
        total_error += abs_error
        total_actual += abs(actual)
        n_points += 1
        predictions[key] = {"actual": actual, "predicted": float(predicted)}
        
        if idx < 10 or idx % 5 == 0:
            rel_error = abs_error / max(abs(actual), 1e-8)
            print(f"  {key}: actual={actual:.4f}, pred={predicted:.4f}, err={abs_error:.4f} ({rel_error:.1%})")
    
    mae = total_error / max(n_points, 1)
    mre = total_error / max(total_actual, 1)
    
    print(f"\n  MAE={mae:.4f}, MRE={mre:.2%}, n={n_points}")
    
    p389_result = {
        "mae": float(mae),
        "mre": float(mre),
        "n_points": n_points,
        "validation_ok": mre < 0.15,
        "predictions": predictions,
    }
    
    ts = time.strftime("%Y%m%d_%H%M", time.localtime())
    out_file = OUT_DIR / f"phase_lxxvi_p389_{model_name}_{ts}.json"
    save_json(p389_result, out_file)
    
    return p389_result


def compute_cos_decay(model, tok, device):
    """Compute cos decay curve."""
    inputs = tok(PROMPT, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    
    W_lm = get_w_lm(model)
    attr = "red"
    attr_id = tok.convert_tokens_to_ids(attr)
    w_lm_attr = W_lm[attr_id].detach()
    w_lm_norm_vec = w_lm_attr / w_lm_attr.norm()
    del W_lm  # Free memory
    torch.cuda.empty_cache()
    
    layers = get_layers(model)
    n_layers = min(len(layers), MAX_LAYERS)
    
    captured_base = {}
    captured_int = {}
    
    def make_hook(captured_dict, prefix):
        def hook_fn(module, input, output, _prefix=prefix):
            if isinstance(output, tuple):
                captured_dict[_prefix] = output[0].detach().cpu()
            else:
                captured_dict[_prefix] = output.detach().cpu()
        return hook_fn
    
    # Base run
    handles = []
    for i in range(n_layers):
        h = layers[i].register_forward_hook(make_hook(captured_base, f"L{i}"))
        handles.append(h)
    
    with torch.no_grad():
        _ = model(input_ids)
    
    for h in handles:
        h.remove()
    
    # Intervention run
    handles = []
    beta = 5.0
    w_lm_device = w_lm_norm_vec.to(device)
    
    for i in range(n_layers):
        if i == 0:
            def inject_and_capture_hook(module, input, output, _i=i):
                if isinstance(output, tuple):
                    h_out = output[0].clone()
                else:
                    h_out = output.clone()
                h_out[:, -1, :] += beta * w_lm_device.to(h_out.dtype)
                captured_int[f"L{_i}"] = h_out.detach().cpu()
                return (h_out,) + output[1:] if isinstance(output, tuple) else h_out
            h = layers[i].register_forward_hook(inject_and_capture_hook)
        else:
            h = layers[i].register_forward_hook(make_hook(captured_int, f"L{i}"))
        handles.append(h)
    
    with torch.no_grad():
        _ = model(input_ids)
    
    for h in handles:
        h.remove()
    
    del w_lm_device
    torch.cuda.empty_cache()
    
    # Compute cos
    cos_data = {}
    w_lm_np = w_lm_attr.numpy()
    
    for key in captured_base.keys():
        if key not in captured_int:
            continue
        h_base = captured_base[key][0, -1, :].float().numpy()
        h_int = captured_int[key][0, -1, :].float().numpy()
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
    
    return cos_data, float(w_lm_attr.norm()), 0.0


def compute_silu_grad(model, tok, device, layer_indices):
    """Compute SiLU gradient statistics."""
    inputs = tok(PROMPT, return_tensors="pt").to(device)
    stats = {}
    
    with torch.no_grad():
        all_layers = get_layers(model)
        for i in layer_indices:
            if i >= len(all_layers):
                continue
            layer = all_layers[i]
            mlp = layer.mlp if hasattr(layer, 'mlp') else layer.feed_forward
            
            is_merged = hasattr(mlp, 'gate_up_proj')
            is_standard = hasattr(mlp, 'gate_proj')
            
            captured = {}
            
            if is_merged:
                def hook_fn_merged(module, input, output, layer_idx=i):
                    gate, _ = output.chunk(2, dim=-1)
                    captured[f"pre_gate_L{layer_idx}"] = gate.detach().cpu()
                
                handle = mlp.gate_up_proj.register_forward_hook(hook_fn_merged)
                try:
                    _ = model(**inputs)
                except:
                    pass
                handle.remove()
            elif is_standard:
                def hook_fn(module, input, output, layer_idx=i):
                    captured[f"pre_gate_L{layer_idx}"] = input[0].detach().cpu()
                
                handle = mlp.gate_proj.register_forward_hook(hook_fn)
                try:
                    _ = model(**inputs)
                except:
                    pass
                handle.remove()
            else:
                continue
            
            if f"pre_gate_L{i}" in captured:
                pre_gate = captured[f"pre_gate_L{i}"][0, -1, :].float()
                sigmoid_x = torch.sigmoid(pre_gate)
                silu_grad = sigmoid_x * (1 + pre_gate * (1 - sigmoid_x))
                stats[f"L{i}"] = {
                    "silu_grad_mean": float(silu_grad.mean()),
                    "silu_grad_std": float(silu_grad.std()),
                    "silu_grad_median": float(silu_grad.median()),
                    "sigmoid_mean": float(sigmoid_x.mean()),
                }
            
            del captured
            torch.cuda.empty_cache()
    
    return stats


def fit_formula(name, fit_layers, cum_sigma_up, w_lm_norm, log_ratio, actual_cos_arr, cos_0, desc):
    """Fit formula a: lambda = c * cum_sigma_up / W_lm_norm."""
    x = np.array([cum_sigma_up[k] / w_lm_norm for k in fit_layers])
    valid = np.isfinite(log_ratio) & np.isfinite(x) & (x > 0)
    if valid.sum() > 0:
        c, _ = np.linalg.lstsq(x[valid].reshape(-1, 1), -log_ratio[valid], rcond=None)[:2]
        c = float(c[0]) if len(c) > 0 else 0.023
        pred_fit = cos_0 * np.exp(-c * x)
        ss_res = np.sum((actual_cos_arr - pred_fit)**2)
        ss_tot = np.sum((actual_cos_arr - actual_cos_arr.mean())**2)
        r2 = 1 - ss_res / max(ss_tot, 1e-10)
    else:
        c, r2 = 0.023, 0.0
    
    print(f"  Formula {name}: c={c:.4f}, R2={r2:.4f}")
    return {"formula": desc, "c1": c, "R2": float(r2)}


def fit_formula_custom(name, fit_layers, x_dict, log_ratio, actual_cos_arr, cos_0, desc):
    """Fit arbitrary formula."""
    x = np.array([x_dict[k] for k in fit_layers])
    valid = np.isfinite(log_ratio) & np.isfinite(x) & (x > 0)
    if valid.sum() > 0:
        c, _ = np.linalg.lstsq(x[valid].reshape(-1, 1), -log_ratio[valid], rcond=None)[:2]
        c = float(c[0]) if len(c) > 0 else 0.0
        pred_fit = cos_0 * np.exp(-c * x)
        ss_res = np.sum((actual_cos_arr - pred_fit)**2)
        ss_tot = np.sum((actual_cos_arr - actual_cos_arr.mean())**2)
        r2 = 1 - ss_res / max(ss_tot, 1e-10)
    else:
        c, r2 = 0.0, 0.0
    
    c_key = f"c{ord(name) - ord('a') + 1}"
    print(f"  Formula {name}: {c_key}={c:.6f}, R2={r2:.4f}")
    return {"formula": desc, c_key: c, "R2": float(r2)}


def save_json(data, filepath):
    """Save with numpy conversion."""
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
    
    data = convert(data)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {filepath}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python phase_lxxvi_glm4_deepseek.py <glm4|deepseek7b>")
        sys.exit(1)
    
    model_name = sys.argv[1]
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}")
        sys.exit(1)
    
    results = {"model": model_name}
    
    try:
        # P387
        p387_data, model, tok, device = run_p387(model_name)
        results["p387"] = p387_data
        
        # P388
        p388_data, model, tok, device = run_p388(model_name, model, tok, device, p387_data)
        results["p388"] = p388_data
        
        # P389
        p389_data = run_p389(model_name, model, tok, device, p387_data, p388_data)
        results["p389"] = p389_data
        
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        results["error"] = str(e)
    finally:
        # Always try to free model
        try:
            free_model(model)
        except:
            pass
    
    # Save combined result
    ts = time.strftime("%Y%m%d_%H%M", time.localtime())
    out_file = OUT_DIR / f"phase_lxxvi_p387_389_{model_name}_{ts}.json"
    save_json(results, out_file)
    
    print(f"\n{'='*60}")
    print(f"COMPLETE: {model_name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
