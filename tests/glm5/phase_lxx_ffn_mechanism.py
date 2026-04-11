"""
Phase LXX-P368/369/370: FFN对注入方向的作用机制分析
======================================================================

核心目标: 破解"为什么Qwen3的FFN吸收注入方向而GLM4的FFN不吸收"

P368: FFN输出方向的精细分析
  - 计算FFN(base)和FFN(intervened)的输出
  - ΔFFN = FFN(intervened) - FFN(base)
  - 关键: cos(ΔFFN, W_lm[attr]) = FFN对注入方向的"响应方向"
  - 如果cos<0 → FFN在"对抗"注入方向
  - 如果cos≈0 → FFN对注入方向"隐形"
  - 如果cos>0 → FFN在"帮助"注入方向
  - 还计算: cos(FFN_base, W_lm[attr]) = FFN基线是否在W_lm方向

P369: FFN权重矩阵的SVD分析
  - W_down (MLP的最后一层) 的奇异值分解
  - 看W_lm[attr]在W_down的奇异向量上的投影
  - 如果W_lm[attr]主要投影在高奇异值向量上 → FFN会强烈响应
  - 如果W_lm[attr]主要投影在低奇异值向量上 → FFN会忽略
  - 对比GLM4和Qwen3的FFN SVD结构差异

P370: FFN中间激活分析
  - FFN(x) = W_down(act(W_up(x) + b_up)) + b_down
  - 分析W_up(x)和act(W_up(x))对注入方向的影响
  - 关键: 如果act(W_up(x))的输出在W_lm方向有分量 → FFN会影响注入方向

实验模型: qwen3 → glm4 → deepseek7b (串行)
"""

import torch
import torch.nn.functional as F
import numpy as np
import os, sys, gc, time, json, argparse, copy
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

import functools, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
print = functools.partial(print, flush=True)

OUT_DIR = Path("tests/glm5_temp")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUT_DIR / "phase_lxx_log.txt"

class Logger:
    def __init__(self, path):
        self.path = path
        self.f = open(path, 'w', encoding='utf-8', buffering=1)
    def log(self, msg):
        ts = time.strftime('%H:%M:%S')
        self.f.write(f"{ts} {msg}\n")
        self.f.flush()
        print(f"  [{ts}] {msg}")
    def close(self): self.f.close()

L = Logger(LOG_FILE)

def get_model_path(model_name):
    paths = {
        "qwen3": r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c",
        "glm4": r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf",
        "deepseek7b": r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60",
    }
    return paths.get(model_name)

def load_model(model_name):
    p = get_model_path(model_name)
    if p is None:
        raise ValueError(f"Unknown model: {model_name}")
    p_abs = os.path.abspath(p)
    tok = AutoTokenizer.from_pretrained(p_abs, trust_remote_code=True, local_files_only=True, use_fast=False)
    tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        p_abs, dtype=torch.bfloat16, trust_remote_code=True,
        local_files_only=True, low_cpu_mem_usage=True,
        attn_implementation="eager", device_map="cpu"
    )
    if torch.cuda.is_available():
        mdl = mdl.to("cuda")
    mdl.eval()
    device = next(mdl.parameters()).device
    return mdl, tok, device

ALL_ATTRS = ["red","green","blue","sweet","sour","hot","cold","big","small","long","wide","soft"]
NOUNS = ["apple","banana","cat","dog","car","bus","chair","table","hammer","wrench"]

def get_attr_direction(model, tokenizer, attr, method="lm_head"):
    attr_tok_ids = tokenizer.encode(attr, add_special_tokens=False)
    if len(attr_tok_ids) == 0:
        return None, None
    if method == "lm_head":
        lm_head = model.lm_head if hasattr(model, 'lm_head') else model.get_output_embeddings()
        direction = lm_head.weight[attr_tok_ids[0]].detach().cpu().float().numpy().copy()
    else:
        embed_layer = model.get_input_embeddings()
        direction = embed_layer.weight[attr_tok_ids[0]].detach().cpu().float().numpy().copy()
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm
    return direction, attr_tok_ids[0]

def make_hook_fn(storage, key):
    def hook(module, input, output):
        if isinstance(output, tuple):
            storage[key] = output[0].detach().float().cpu()
        else:
            storage[key] = output.detach().float().cpu()
    return hook


# ===================== P368: FFN输出方向精细分析 =====================
def run_p368(model, tokenizer, device, model_name):
    """
    ★★★ 核心实验: FFN对注入方向产生什么方向的输出? ★★★
    
    计算:
    - ΔFFN = FFN(intervened_input) - FFN(base_input)
    - cos(ΔFFN, W_lm[attr]) = FFN响应的"方向"
    - cos(FFN_base, W_lm[attr]) = FFN基线输出是否在W_lm方向
    - ||ΔFFN|| / ||Δh_input|| = FFN响应的"强度"相对于输入变化
    """
    L.log("=== P368: FFN输出方向精细分析 ===")
    
    beta = 8.0
    n_layers = len(model.model.layers) if hasattr(model, 'model') and hasattr(model.model, 'layers') else 0
    D = model.config.hidden_size
    
    if n_layers == 0:
        L.log("  无Transformer层, 跳过")
        return {}
    
    layer0 = model.model.layers[0]
    if not hasattr(layer0, 'mlp'):
        L.log("  无mlp, 跳过")
        return {}
    
    # 检查FFN结构
    mlp = layer0.mlp
    L.log(f"  MLP类型: {type(mlp).__name__}")
    for name, param in mlp.named_parameters():
        L.log(f"    {name}: shape={list(param.shape)}")
    
    test_attrs = ["red", "sweet", "hot", "big"]
    results = {}
    
    for ai, attr in enumerate(test_attrs):
        direction, attr_tok_id = get_attr_direction(model, tokenizer, attr, "lm_head")
        if direction is None:
            continue
        
        L.log(f"  [{ai+1}/4] {attr}")
        
        lm_head = model.lm_head if hasattr(model, 'lm_head') else model.get_output_embeddings()
        w_lm_attr = lm_head.weight[attr_tok_id].detach().cpu().float()
        w_lm_attr_norm = w_lm_attr / w_lm_attr.norm()
        w_lm_np = w_lm_attr_norm.numpy()
        
        attr_data = []
        
        for noun in NOUNS[:3]:
            prompt = f"The {noun} is"
            toks = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = toks.input_ids
            seq_len = input_ids.shape[1]
            embed_layer = model.get_input_embeddings()
            inputs_embeds_base = embed_layer(input_ids).clone()
            
            direction_tensor = torch.tensor(direction, dtype=inputs_embeds_base.dtype, device=device)
            inputs_embeds_intervened = inputs_embeds_base.clone()
            inputs_embeds_intervened[0, -1, :] += beta * direction_tensor
            
            # 收集FFN输出
            captured_base = {}
            captured_interv = {}
            
            hooks_base = []
            hooks_interv = []
            
            # Hook: FFN输出
            hooks_base.append(layer0.mlp.register_forward_hook(
                make_hook_fn(captured_base, "ffn_out")))
            # Hook: Attn输出
            if hasattr(layer0, 'self_attn'):
                hooks_base.append(layer0.self_attn.register_forward_hook(
                    make_hook_fn(captured_base, "attn_out")))
            # Hook: Layer整体输出
            hooks_base.append(layer0.register_forward_hook(
                make_hook_fn(captured_base, "layer_out")))
            # Hook: FFN输入 (post_attention_layernorm的输出)
            if hasattr(layer0, 'post_attention_layernorm'):
                hooks_base.append(layer0.post_attention_layernorm.register_forward_hook(
                    make_hook_fn(captured_base, "ffn_input")))
            
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            
            with torch.no_grad():
                try:
                    _ = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids)
                except:
                    for h in hooks_base: h.remove()
                    continue
            for h in hooks_base: h.remove()
            
            # Intervened
            hooks_interv.append(layer0.mlp.register_forward_hook(
                make_hook_fn(captured_interv, "ffn_out")))
            if hasattr(layer0, 'self_attn'):
                hooks_interv.append(layer0.self_attn.register_forward_hook(
                    make_hook_fn(captured_interv, "attn_out")))
            hooks_interv.append(layer0.register_forward_hook(
                make_hook_fn(captured_interv, "layer_out")))
            if hasattr(layer0, 'post_attention_layernorm'):
                hooks_interv.append(layer0.post_attention_layernorm.register_forward_hook(
                    make_hook_fn(captured_interv, "ffn_input")))
            
            with torch.no_grad():
                try:
                    _ = model(inputs_embeds=inputs_embeds_intervened, position_ids=position_ids)
                except:
                    for h in hooks_interv: h.remove()
                    continue
            for h in hooks_interv: h.remove()
            
            # 分析FFN的响应
            analysis = {}
            
            for comp in ["ffn_out", "attn_out", "layer_out", "ffn_input"]:
                key_b = comp
                key_i = comp
                if key_b not in captured_base or key_i not in captured_interv:
                    continue
                
                h_base = captured_base[key_b][0, -1, :].numpy()
                h_interv = captured_interv[key_i][0, -1, :].numpy()
                delta = h_interv - h_base
                delta_norm = np.linalg.norm(delta)
                
                # cos(Δ, W_lm[attr])
                cos_wlm = float(np.dot(delta, w_lm_np) / delta_norm) if delta_norm > 1e-8 else 0.0
                
                # cos(基线输出, W_lm[attr])
                base_norm = np.linalg.norm(h_base)
                cos_base_wlm = float(np.dot(h_base, w_lm_np) / base_norm) if base_norm > 1e-8 else 0.0
                
                # Δ的范数
                analysis[comp] = {
                    "cos_delta_wlm": round(cos_wlm, 6),
                    "cos_base_wlm": round(cos_base_wlm, 6),
                    "delta_norm": round(delta_norm, 4),
                    "base_norm": round(base_norm, 4),
                    "proj_delta_on_wlm": round(float(np.dot(delta, w_lm_np)), 6),
                }
            
            attr_data.append({"noun": noun, "analysis": analysis})
        
        # 汇总
        avg = {}
        for comp in ["ffn_out", "attn_out", "layer_out", "ffn_input"]:
            cos_deltas = [d["analysis"][comp]["cos_delta_wlm"] for d in attr_data if comp in d["analysis"]]
            cos_bases = [d["analysis"][comp]["cos_base_wlm"] for d in attr_data if comp in d["analysis"]]
            proj_deltas = [d["analysis"][comp]["proj_delta_on_wlm"] for d in attr_data if comp in d["analysis"]]
            if cos_deltas:
                avg[comp] = {
                    "cos_delta_wlm": round(np.mean(cos_deltas), 6),
                    "cos_base_wlm": round(np.mean(cos_bases), 6),
                    "avg_proj_delta_on_wlm": round(np.mean(proj_deltas), 6),
                }
        
        results[attr] = avg
        for comp, vals in avg.items():
            L.log(f"    {comp}: cos(ΔFFN,W_lm)={vals['cos_delta_wlm']}, "
                  f"cos(FFN_base,W_lm)={vals['cos_base_wlm']}, "
                  f"proj_Δ_on_wlm={vals['avg_proj_delta_on_wlm']}")
    
    return results


# ===================== P369: FFN权重矩阵SVD分析 =====================
def run_p369(model, tokenizer, device, model_name):
    """
    ★★★ FFN的权重矩阵是否"回避"W_lm方向? ★★★
    
    FFN(x) = W_down(act(W_up(x)))
    
    W_down: [D, intermediate_size] → 输出投影
    W_up: [intermediate_size, D] → 输入投影
    
    关键问题:
    - W_down的行向量(每个输出维度)是否正交于W_lm[attr]?
    - W_up的列向量(每个输入维度的投影)是否正交于W_lm[attr]?
    
    如果W_down正交于W_lm → FFN输出不在W_lm方向 → "隐形"
    如果W_down不正交于W_lm → FFN输出在W_lm方向 → "干扰"
    """
    L.log("=== P369: FFN权重矩阵SVD分析 ===")
    
    n_layers = len(model.model.layers) if hasattr(model, 'model') and hasattr(model.model, 'layers') else 0
    D = model.config.hidden_size
    
    if n_layers == 0:
        L.log("  无Transformer层, 跳过")
        return {}
    
    layer0 = model.model.layers[0]
    if not hasattr(layer0, 'mlp'):
        L.log("  无mlp, 跳过")
        return {}
    
    mlp = layer0.mlp
    
    # 获取权重矩阵
    # 常见结构: gate_proj, up_proj, down_proj (SwiGLU)
    # 或: W_up, W_down (标准MLP)
    W_down = None
    W_up = None
    W_gate = None
    
    if hasattr(mlp, 'down_proj'):
        W_down = mlp.down_proj.weight.detach().cpu().float()  # [D, intermediate]
        L.log(f"  down_proj: shape={list(W_down.shape)}")
    if hasattr(mlp, 'up_proj'):
        W_up = mlp.up_proj.weight.detach().cpu().float()  # [intermediate, D]
        L.log(f"  up_proj: shape={list(W_up.shape)}")
    if hasattr(mlp, 'gate_proj'):
        W_gate = mlp.gate_proj.weight.detach().cpu().float()  # [intermediate, D]
        L.log(f"  gate_proj: shape={list(W_gate.shape)}")
    
    results = {}
    
    test_attrs = ["red", "sweet", "hot", "big"]
    
    for ai, attr in enumerate(test_attrs):
        direction, attr_tok_id = get_attr_direction(model, tokenizer, attr, "lm_head")
        if direction is None:
            continue
        
        L.log(f"  [{ai+1}/4] {attr}")
        
        lm_head = model.lm_head if hasattr(model, 'lm_head') else model.get_output_embeddings()
        w_lm_attr = lm_head.weight[attr_tok_id].detach().cpu().float()
        w_lm_np = (w_lm_attr / w_lm_attr.norm()).numpy()
        
        attr_result = {}
        
        if W_down is not None:
            # W_down的每一行是输出维度
            # cos(W_down[i], w_lm) for each i
            # W_down: [D, intermediate] → 每行长度=intermediate
            # 但w_lm是D维的, W_down的输出也是D维的!
            # 不对: W_down将intermediate→D, 所以W_down的行是D维的
            # 等等: W_down.weight shape=[D, intermediate]
            # 所以W_down[i]是D维的! 可以与w_lm比较!
            
            # 计算每个输出维度与w_lm的对齐度
            W_down_np = W_down.numpy()  # [D, intermediate]
            
            # 逐行计算cos(W_down[i], w_lm)
            cos_per_row = []
            for i in range(W_down_np.shape[0]):
                row = W_down_np[i, :]
                # 不对, W_down[i]是[intermediate], 不是D维
                # W_down的输出 = W_down @ x, 其中x是intermediate维
                # 所以W_down的输出维度是D
                # 但W_down的行向量是intermediate维的
                # 要看"输出空间"的对齐, 应该看W_down^T的行(=W_down的列)
                pass
            
            # 正确的做法: FFN输出 = W_down @ h_intermediate
            # 其中h_intermediate = act(W_up @ x)
            # FFN输出[i] = sum_j W_down[i,j] * h_intermediate[j]
            # 所以FFN输出方向 = W_down的行的线性组合
            
            # 关键问题: W_down能否产生w_lm方向的输出?
            # 即: 是否存在h_intermediate使得W_down @ h_intermediate ∝ w_lm?
            # 这等价于: w_lm是否在W_down的列空间中?
            
            # W_down: [D, intermediate], 列空间是D维空间中由intermediate个列张成的子空间
            # 如果intermediate > D (通常如此), 则列空间就是整个D维空间!
            # → W_down总是能产生w_lm方向的输出!
            
            # 所以问题不在于W_down能否产生w_lm, 而在于:
            # 在正常输入下, h_intermediate的值使得FFN输出恰好有多少在w_lm方向
            
            # 另一种分析: W_down^T @ w_lm = "w_lm在W_down行空间中的投影"
            # 即: 如果W_down^T @ w_lm很大 → W_down的行向量"覆盖"了w_lm方向
            
            # 更直接的分析: 
            # FFN输出在w_lm方向的投影 = w_lm^T @ (W_down @ h_intermediate)
            #                           = (W_down^T @ w_lm)^T @ h_intermediate
            # 其中W_down^T @ w_lm是"FFN中间表示对w_lm方向的敏感度"
            
            W_down_T_wlm = W_down.T.numpy() @ w_lm_np  # [intermediate]
            # 这是w_lm在W_down列空间中的"投影坐标"
            
            attr_result["W_down_T_wlm_norm"] = round(float(np.linalg.norm(W_down_T_wlm)), 4)
            attr_result["W_down_T_wlm_mean"] = round(float(np.mean(np.abs(W_down_T_wlm))), 6)
            attr_result["W_down_T_wlm_max"] = round(float(np.max(np.abs(W_down_T_wlm))), 6)
            
            # 也计算: W_down的SVD
            # 取前10个奇异值
            L.log(f"    计算W_down SVD...")
            try:
                # W_down: [D, intermediate], 太大无法完整SVD
                # 只计算前10个奇异值
                U, S, Vt = np.linalg.svd(W_down_np, full_matrices=False)
                
                # w_lm在U的各奇异向量上的投影
                proj_on_U = U.T @ w_lm_np  # [D] → 投影到每个奇异向量
                
                # 前10个奇异值和投影
                top_k = min(20, len(S))
                attr_result["SVD"] = {
                    "top_singular_values": [round(float(s), 4) for s in S[:top_k]],
                    "wlm_proj_on_top_U": [round(float(p), 6) for p in proj_on_U[:top_k]],
                    "wlm_proj_explained_ratio": round(
                        float(np.sum(proj_on_U[:top_k]**2) / np.sum(proj_on_U**2)), 4),
                }
                L.log(f"    W_down SVD: top5 S={[round(float(s),2) for s in S[:5]]}, "
                      f"wlm在top5投影占比={attr_result['SVD']['wlm_proj_explained_ratio']}")
            except Exception as e:
                L.log(f"    SVD失败: {e}")
                attr_result["SVD"] = str(e)
        
        if W_up is not None:
            # W_up: [intermediate, D]
            # W_up @ x = 输入x在intermediate维空间的投影
            # W_up @ w_lm = w_lm在FFN输入空间的"映射"
            W_up_np = W_up.numpy()
            W_up_wlm = W_up_np @ w_lm_np  # [intermediate]
            
            attr_result["W_up_wlm_norm"] = round(float(np.linalg.norm(W_up_wlm)), 4)
            attr_result["W_up_wlm_mean"] = round(float(np.mean(np.abs(W_up_wlm))), 6)
            
            L.log(f"    W_up @ w_lm: norm={attr_result['W_up_wlm_norm']}, "
                  f"mean_abs={attr_result['W_up_wlm_mean']}")
        
        if W_gate is not None:
            W_gate_np = W_gate.numpy()
            W_gate_wlm = W_gate_np @ w_lm_np  # [intermediate]
            
            attr_result["W_gate_wlm_norm"] = round(float(np.linalg.norm(W_gate_wlm)), 4)
            attr_result["W_gate_wlm_mean"] = round(float(np.mean(np.abs(W_gate_wlm))), 6)
            
            L.log(f"    W_gate @ w_lm: norm={attr_result['W_gate_wlm_norm']}, "
                  f"mean_abs={attr_result['W_gate_wlm_mean']}")
        
        results[attr] = attr_result
    
    # 释放大矩阵
    del W_down, W_up
    if W_gate is not None:
        del W_gate
    gc.collect()
    
    return results


# ===================== P370: FFN中间激活分析 =====================
def run_p370(model, tokenizer, device, model_name):
    """
    ★★★ FFN内部各层如何响应注入方向? ★★★
    
    FFN(x) = W_down(act(W_gate(x)) * W_up(x))  (SwiGLU)
    或 FFN(x) = W_down(act(W_up(x)))  (标准)
    
    关键: 在intervened时, FFN的输入是"被RMSNorm和Attn修改过的"
    所以要跟踪:
    1. FFN输入 = post_attn_layernorm(h + attn_out)
    2. FFN输出 = W_down(...)
    3. ΔFFN_input = FFN_input_interv - FFN_input_base
    4. cos(ΔFFN_input, W_lm[attr]) = FFN输入对注入方向的响应
    """
    L.log("=== P370: FFN中间激活分析 ===")
    
    beta = 8.0
    n_layers = len(model.model.layers) if hasattr(model, 'model') and hasattr(model.model, 'layers') else 0
    
    if n_layers == 0:
        L.log("  无Transformer层, 跳过")
        return {}
    
    layer0 = model.model.layers[0]
    
    test_attrs = ["red", "sweet"]
    results = {}
    
    for ai, attr in enumerate(test_attrs):
        direction, attr_tok_id = get_attr_direction(model, tokenizer, attr, "lm_head")
        if direction is None:
            continue
        
        L.log(f"  [{ai+1}/2] {attr}")
        
        lm_head = model.lm_head if hasattr(model, 'lm_head') else model.get_output_embeddings()
        w_lm_attr = lm_head.weight[attr_tok_id].detach().cpu().float()
        w_lm_attr_norm = w_lm_attr / w_lm_attr.norm()
        w_lm_np = w_lm_attr_norm.numpy()
        
        noun = "apple"
        prompt = f"The {noun} is"
        toks = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = toks.input_ids
        seq_len = input_ids.shape[1]
        embed_layer = model.get_input_embeddings()
        inputs_embeds_base = embed_layer(input_ids).clone()
        
        direction_tensor = torch.tensor(direction, dtype=inputs_embeds_base.dtype, device=device)
        inputs_embeds_intervened = inputs_embeds_base.clone()
        inputs_embeds_intervened[0, -1, :] += beta * direction_tensor
        
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        
        # 收集所有关键节点的hidden state
        captured_base = {}
        captured_interv = {}
        
        # === Base forward ===
        hooks_base = []
        if hasattr(layer0, 'input_layernorm'):
            hooks_base.append(layer0.input_layernorm.register_forward_hook(
                make_hook_fn(captured_base, "input_ln")))
        if hasattr(layer0, 'self_attn'):
            hooks_base.append(layer0.self_attn.register_forward_hook(
                make_hook_fn(captured_base, "attn_out")))
        if hasattr(layer0, 'post_attention_layernorm'):
            hooks_base.append(layer0.post_attention_layernorm.register_forward_hook(
                make_hook_fn(captured_base, "post_attn_ln")))
        if hasattr(layer0, 'mlp'):
            hooks_base.append(layer0.mlp.register_forward_hook(
                make_hook_fn(captured_base, "mlp_out")))
        hooks_base.append(layer0.register_forward_hook(
            make_hook_fn(captured_base, "layer_out")))
        
        with torch.no_grad():
            try:
                _ = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids)
            except:
                for h in hooks_base: h.remove()
                continue
        for h in hooks_base: h.remove()
        
        # === Intervened forward ===
        hooks_interv = []
        if hasattr(layer0, 'input_layernorm'):
            hooks_interv.append(layer0.input_layernorm.register_forward_hook(
                make_hook_fn(captured_interv, "input_ln")))
        if hasattr(layer0, 'self_attn'):
            hooks_interv.append(layer0.self_attn.register_forward_hook(
                make_hook_fn(captured_interv, "attn_out")))
        if hasattr(layer0, 'post_attention_layernorm'):
            hooks_interv.append(layer0.post_attention_layernorm.register_forward_hook(
                make_hook_fn(captured_interv, "post_attn_ln")))
        if hasattr(layer0, 'mlp'):
            hooks_interv.append(layer0.mlp.register_forward_hook(
                make_hook_fn(captured_interv, "mlp_out")))
        hooks_interv.append(layer0.register_forward_hook(
            make_hook_fn(captured_interv, "layer_out")))
        
        with torch.no_grad():
            try:
                _ = model(inputs_embeds=inputs_embeds_intervened, position_ids=position_ids)
            except:
                for h in hooks_interv: h.remove()
                continue
        for h in hooks_interv: h.remove()
        
        # 分析各节点
        nodes = ["input_ln", "attn_out", "post_attn_ln", "mlp_out", "layer_out"]
        node_analysis = {}
        
        for node in nodes:
            if node not in captured_base or node not in captured_interv:
                continue
            
            h_base = captured_base[node][0, -1, :].numpy()
            h_interv = captured_interv[node][0, -1, :].numpy()
            delta = h_interv - h_base
            delta_norm = np.linalg.norm(delta)
            base_norm = np.linalg.norm(h_base)
            
            cos_delta_wlm = float(np.dot(delta, w_lm_np) / delta_norm) if delta_norm > 1e-8 else 0.0
            cos_base_wlm = float(np.dot(h_base, w_lm_np) / base_norm) if base_norm > 1e-8 else 0.0
            proj_delta = float(np.dot(delta, w_lm_np))
            
            node_analysis[node] = {
                "cos_delta_wlm": round(cos_delta_wlm, 6),
                "cos_base_wlm": round(cos_base_wlm, 6),
                "delta_norm": round(delta_norm, 4),
                "base_norm": round(base_norm, 4),
                "proj_delta_on_wlm": round(proj_delta, 6),
            }
        
        results[attr] = node_analysis
        
        for node, vals in node_analysis.items():
            L.log(f"    {node}: cos(Δ,W_lm)={vals['cos_delta_wlm']}, "
                  f"cos(base,W_lm)={vals['cos_base_wlm']}, "
                  f"||Δ||={vals['delta_norm']}, proj_Δ={vals['proj_delta_on_wlm']}")
        
        # 关键分析: 信号在各节点的"衰减"
        # proj_delta_on_wlm表示"注入方向在Δ中的投影"
        # 如果proj在post_attn_ln处比input_ln处小 → Attn在"吸收"方向
        # 如果proj在mlp_out处比post_attn_ln处小 → MLP在"吸收"方向
        
        if "input_ln" in node_analysis and "post_attn_ln" in node_analysis and "mlp_out" in node_analysis:
            proj_input = abs(node_analysis["input_ln"]["proj_delta_on_wlm"])
            proj_post_attn = abs(node_analysis["post_attn_ln"]["proj_delta_on_wlm"])
            proj_mlp = abs(node_analysis["mlp_out"]["proj_delta_on_wlm"])
            proj_layer = abs(node_analysis["layer_out"]["proj_delta_on_wlm"])
            
            L.log(f"    ★ 信号流向: input_ln={proj_input:.4f} → "
                  f"post_attn_ln={proj_post_attn:.4f} → "
                  f"mlp_out={proj_mlp:.4f} → "
                  f"layer_out={proj_layer:.4f}")
            
            if proj_post_attn < proj_input:
                L.log(f"    ★ Attn吸收了{(1-proj_post_attn/proj_input)*100:.1f}%的W_lm投影!")
            if proj_mlp < proj_post_attn:
                L.log(f"    ★ MLP吸收了{(1-proj_mlp/proj_post_attn)*100:.1f}%的W_lm投影!")
    
    return results


# ===================== 主函数 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "p368", "p369", "p370"])
    args = parser.parse_args()
    
    model_name = args.model
    L.log(f"===== Phase LXX: FFN对注入方向的作用机制分析 =====")
    L.log(f"模型: {model_name}")
    
    L.log("加载模型...")
    model, tokenizer, device = load_model(model_name)
    n_layers = len(model.model.layers) if hasattr(model, 'model') and hasattr(model.model, 'layers') else 0
    D = model.config.hidden_size
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    
    L.log(f"模型: {model_name}, 层数: {n_layers}, 维度: {D}, 参数量: {n_params:.1f}B")
    
    all_results = {
        "model": model_name,
        "n_layers": n_layers,
        "D": D,
        "n_params_B": round(n_params, 2),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    if args.experiment in ["all", "p368"]:
        L.log("P368: FFN输出方向精细分析...")
        all_results["p368"] = run_p368(model, tokenizer, device, model_name)
    
    if args.experiment in ["all", "p369"]:
        L.log("P369: FFN权重矩阵SVD分析...")
        all_results["p369"] = run_p369(model, tokenizer, device, model_name)
    
    if args.experiment in ["all", "p370"]:
        L.log("P370: FFN中间激活分析...")
        all_results["p370"] = run_p370(model, tokenizer, device, model_name)
    
    # 保存结果
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = OUT_DIR / f"phase_lxx_p368_370_{model_name}_{ts}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    L.log(f"结果已保存: {out_path}")
    
    # 释放GPU
    del model
    gc.collect()
    torch.cuda.empty_cache()
    L.log("GPU已释放")
    L.close()

if __name__ == "__main__":
    main()
