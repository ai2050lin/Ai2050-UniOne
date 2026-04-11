"""
Phase LXVIII-P362/363/364: L0方向对齐度根源分析
======================================================================

核心目标: 破解"为什么GLM4的L0方向对齐度=0.999而Qwen3只有0.60"

P362: ★★★最核心★★★ LayerNorm/RMSNorm对注入方向的影响
  - 在embedding后、第1层前, 加一个hook看RMSNorm如何改变注入方向
  - 计算RMSNorm前后的 cos(Δh, W_lm[attr])
  - 如果RMSNorm后cos下降 → RMSNorm是方向扭曲的来源
  - 如果RMSNorm后cos不下降 → 扭曲在后续层

P363: 注意力偏置和GQA的影响
  - GLM4: attention_bias=true, num_kv_heads=2(极端GQA)
  - Qwen3: attention_bias=false, num_kv_heads=8
  - DS7B: attention_bias=false, num_kv_heads=4
  - 分析第1层注意力如何处理注入方向

P364: "token位置"vs"注入方向"的交互
  - 在不同位置(0,1,...,seq_len-1)注入方向
  - 看L0方向对齐度是否与位置有关
  - 如果在特定位置注入时cos特别高/低 → 位置编码的影响

实验模型: qwen3 → glm4 → deepseek7b (串行)
"""

import torch
import torch.nn.functional as F
import numpy as np
import os, sys, gc, time, json, argparse
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
LOG_FILE = OUT_DIR / "phase_lxviii_log.txt"

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

# ===================== P362: RMSNorm对注入方向的影响 =====================
def run_p362(model, tokenizer, device, model_name):
    """
    ★★★ 核心实验: RMSNorm是否扭曲注入方向? ★★★
    
    思路: 
    1. 在embedding上加β·W_lm[attr]
    2. 在第1层Transformer的RMSNorm前后分别记录hidden state
    3. 看RMSNorm如何改变Δh
    
    RMSNorm(x) = x / sqrt(mean(x²) + eps) * γ
    
    关键: RMSNorm是逐元素的缩放, 如果x包含了额外的方向分量,
    缩放因子会改变, 从而改变方向的相对比例
    """
    L.log("=== P362: RMSNorm对注入方向的影响 ===")
    
    beta = 8.0
    n_layers = len(model.model.layers) if hasattr(model, 'model') and hasattr(model.model, 'layers') else 0
    D = model.config.hidden_size
    
    # 检查模型的norm层位置
    # 大多数模型结构: embed → (可选的embed_norm) → layers[0].input_layernorm → attention → ...
    
    results = {}
    
    for ai, attr in enumerate(ALL_ATTRS):
        direction, attr_tok_id = get_attr_direction(model, tokenizer, attr, "lm_head")
        if direction is None:
            continue
        
        L.log(f"  [{ai+1}/12] {attr}")
        
        lm_head = model.lm_head if hasattr(model, 'lm_head') else model.get_output_embeddings()
        w_lm_attr = lm_head.weight[attr_tok_id].detach().cpu().float()
        w_lm_attr_norm = w_lm_attr / w_lm_attr.norm()
        w_lm_np = w_lm_attr_norm.numpy()
        
        attr_data = []
        
        for noun in NOUNS[:5]:
            prompt = f"The {noun} is"
            toks = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = toks.input_ids
            seq_len = input_ids.shape[1]
            
            embed_layer = model.get_input_embeddings()
            inputs_embeds_base = embed_layer(input_ids).clone()
            
            direction_tensor = torch.tensor(direction, dtype=inputs_embeds_base.dtype, device=device)
            inputs_embeds_intervened = inputs_embeds_base.clone()
            inputs_embeds_intervened[0, -1, :] += beta * direction_tensor
            
            # Hook收集关键节点的hidden state
            captured = {}
            
            def make_hook(storage, key):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        storage[key] = output[0].detach().float().cpu()
                    else:
                        storage[key] = output.detach().float().cpu()
                return hook
            
            # 收集base的各层hidden state
            hooks = []
            
            # Hook到model的embed_norm(如果有)
            if hasattr(model, 'model') and hasattr(model.model, 'embed_norm'):
                hooks.append(model.model.embed_norm.register_forward_hook(make_hook(captured, "embed_norm_base")))
            
            # Hook到第1层的input_layernorm的输入和输出
            if hasattr(model, 'model') and hasattr(model.model, 'layers') and n_layers > 0:
                layer0 = model.model.layers[0]
                if hasattr(layer0, 'input_layernorm'):
                    # Hook到input_layernorm的输出
                    hooks.append(layer0.input_layernorm.register_forward_hook(
                        make_hook(captured, "L0_input_norm_base")))
                
                # Hook到layer0的整体输出
                hooks.append(layer0.register_forward_hook(make_hook(captured, "L0_output_base")))
            
            # Hook到第2层(如果存在)
            if n_layers > 1:
                hooks.append(model.model.layers[1].register_forward_hook(
                    make_hook(captured, "L1_output_base")))
            
            # Base forward
            with torch.no_grad():
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                try:
                    _ = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids)
                except:
                    for h in hooks: h.remove()
                    continue
            for h in hooks: h.remove()
            
            # Intervened forward
            captured_interv = {}
            hooks2 = []
            
            if hasattr(model, 'model') and hasattr(model.model, 'embed_norm'):
                hooks2.append(model.model.embed_norm.register_forward_hook(
                    make_hook(captured_interv, "embed_norm_interv")))
            
            if hasattr(model, 'model') and hasattr(model.model, 'layers') and n_layers > 0:
                layer0 = model.model.layers[0]
                if hasattr(layer0, 'input_layernorm'):
                    hooks2.append(layer0.input_layernorm.register_forward_hook(
                        make_hook(captured_interv, "L0_input_norm_interv")))
                hooks2.append(layer0.register_forward_hook(make_hook(captured_interv, "L0_output_interv")))
            
            if n_layers > 1:
                hooks2.append(model.model.layers[1].register_forward_hook(
                    make_hook(captured_interv, "L1_output_interv")))
            
            with torch.no_grad():
                try:
                    _ = model(inputs_embeds=inputs_embeds_intervened, position_ids=position_ids)
                except:
                    for h in hooks2: h.remove()
                    continue
            for h in hooks2: h.remove()
            
            # 分析各节点的方向保持度
            nodes = ["embed_norm", "L0_input_norm", "L0_output", "L1_output"]
            node_data = {}
            
            for node in nodes:
                key_base = f"{node}_base"
                key_interv = f"{node}_interv"
                if key_base not in captured or key_interv not in captured_interv:
                    continue
                
                h_base = captured[key_base][0, -1, :].numpy()
                h_interv = captured_interv[key_interv][0, -1, :].numpy()
                delta_h = h_interv - h_base
                
                delta_norm = np.linalg.norm(delta_h)
                if delta_norm > 1e-8:
                    cos_val = float(np.dot(delta_h, w_lm_np) / delta_norm)
                else:
                    cos_val = 0.0
                
                proj_len = float(np.dot(delta_h, w_lm_np))
                
                node_data[node] = {
                    "cos": round(cos_val, 4),
                    "delta_norm": round(delta_norm, 4),
                    "proj_on_attr": round(proj_len, 4),
                }
            
            # 还计算"注入瞬间"的方向保持度
            # 即: Δh_embed = inputs_embeds_intervened[-1] - inputs_embeds_base[-1] = beta * direction
            delta_embed = (inputs_embeds_intervened[0, -1, :] - inputs_embeds_base[0, -1, :]).detach().cpu().float().numpy()
            delta_embed_norm = np.linalg.norm(delta_embed)
            if delta_embed_norm > 1e-8:
                cos_embed = float(np.dot(delta_embed, w_lm_np) / delta_embed_norm)
            else:
                cos_embed = 0.0
            
            node_data["raw_embed"] = {
                "cos": round(cos_embed, 4),
                "delta_norm": round(delta_embed_norm, 4),
                "proj_on_attr": round(float(np.dot(delta_embed, w_lm_np)), 4),
            }
            
            attr_data.append({"noun": noun, "nodes": node_data})
        
        # 汇总
        avg_cos = {}
        for node in ["raw_embed", "embed_norm", "L0_input_norm", "L0_output", "L1_output"]:
            cos_vals = [d["nodes"][node]["cos"] for d in attr_data if node in d["nodes"]]
            if cos_vals:
                avg_cos[node] = round(np.mean(cos_vals), 4)
        
        results[attr] = avg_cos
        cos_str = ", ".join([f"{k}={v}" for k, v in avg_cos.items()])
        L.log(f"    {cos_str}")
    
    return results

# ===================== P363: 注意力偏置和GQA影响 =====================
def run_p363(model, tokenizer, device, model_name):
    """
    分析第1层注意力的输出, 看注意力如何处理注入方向
    
    h_L0 = h_input + Attn_out + FFN_out
    Δh_L0 = Δh_input + ΔAttn_out + ΔFFN_out
    
    可以分别看Attention和FFN对方向的影响
    """
    L.log("=== P363: 第1层Attention和FFN对方向的影响 ===")
    
    beta = 8.0
    n_layers = len(model.model.layers) if hasattr(model, 'model') and hasattr(model.model, 'layers') else 0
    D = model.config.hidden_size
    
    test_attrs = ["red","green","sweet","sour","hot","cold","big","small"]
    
    results = {}
    
    for ai, attr in enumerate(test_attrs):
        direction, attr_tok_id = get_attr_direction(model, tokenizer, attr, "lm_head")
        if direction is None:
            continue
        
        L.log(f"  [{ai+1}/8] {attr}")
        
        lm_head = model.lm_head if hasattr(model, 'lm_head') else model.get_output_embeddings()
        w_lm_attr = lm_head.weight[attr_tok_id].detach().cpu().float()
        w_lm_attr_norm = w_lm_attr / w_lm_attr.norm()
        w_lm_np = w_lm_attr_norm.numpy()
        
        # 收集第1层的attn_out和ffn_out
        # h_L0 = h_input + Attn_out + FFN_out (残差连接)
        # 但直接获取attn_out需要hook到self_attn的输出
        
        attr_data = []
        
        for noun in NOUNS[:5]:
            prompt = f"The {noun} is"
            toks = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = toks.input_ids
            seq_len = input_ids.shape[1]
            
            embed_layer = model.get_input_embeddings()
            inputs_embeds_base = embed_layer(input_ids).clone()
            
            direction_tensor = torch.tensor(direction, dtype=inputs_embeds_base.dtype, device=device)
            inputs_embeds_intervened = inputs_embeds_base.clone()
            inputs_embeds_intervened[0, -1, :] += beta * direction_tensor
            
            # Hook收集base的layer0: self_attn输出, mlp输出, 整体输出
            captured_base = {}
            hooks = []
            if n_layers > 0:
                layer0 = model.model.layers[0]
                # self_attn输出 (残差连接前)
                if hasattr(layer0, 'self_attn'):
                    hooks.append(layer0.self_attn.register_forward_hook(
                        make_hook_fn(captured_base, "attn_base")))
                # mlp输出 (残差连接前)
                if hasattr(layer0, 'mlp'):
                    hooks.append(layer0.mlp.register_forward_hook(
                        make_hook_fn(captured_base, "ffn_base")))
                # 整体输出
                hooks.append(layer0.register_forward_hook(
                    make_hook_fn(captured_base, "layer_base")))
            
            with torch.no_grad():
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                try:
                    _ = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids)
                except:
                    for h in hooks: h.remove()
                    continue
            for h in hooks: h.remove()
            
            # Intervened
            captured_interv = {}
            hooks2 = []
            if n_layers > 0:
                layer0 = model.model.layers[0]
                if hasattr(layer0, 'self_attn'):
                    hooks2.append(layer0.self_attn.register_forward_hook(
                        make_hook_fn(captured_interv, "attn_interv")))
                if hasattr(layer0, 'mlp'):
                    hooks2.append(layer0.mlp.register_forward_hook(
                        make_hook_fn(captured_interv, "ffn_interv")))
                hooks2.append(layer0.register_forward_hook(
                    make_hook_fn(captured_interv, "layer_interv")))
            
            with torch.no_grad():
                try:
                    _ = model(inputs_embeds=inputs_embeds_intervened, position_ids=position_ids)
                except:
                    for h in hooks2: h.remove()
                    continue
            for h in hooks2: h.remove()
            
            # 分析各组件对方向的贡献
            components = {}
            for comp in ["attn", "ffn", "layer"]:
                key_b = f"{comp}_base"
                key_i = f"{comp}_interv"
                if key_b not in captured_base or key_i not in captured_interv:
                    continue
                
                h_base = captured_base[key_b][0, -1, :].numpy()
                h_interv = captured_interv[key_i][0, -1, :].numpy()
                delta = h_interv - h_base
                
                delta_norm = np.linalg.norm(delta)
                if delta_norm > 1e-8:
                    cos_val = float(np.dot(delta, w_lm_np) / delta_norm)
                else:
                    cos_val = 0.0
                
                proj = float(np.dot(delta, w_lm_np))
                
                components[comp] = {
                    "cos": round(cos_val, 4),
                    "delta_norm": round(delta_norm, 4),
                    "proj": round(proj, 4),
                }
            
            attr_data.append({"noun": noun, "components": components})
        
        # 汇总
        avg_cos = {}
        for comp in ["attn", "ffn", "layer"]:
            cos_vals = [d["components"][comp]["cos"] for d in attr_data if comp in d["components"]]
            if cos_vals:
                avg_cos[comp] = round(np.mean(cos_vals), 4)
        
        results[attr] = avg_cos
        cos_str = ", ".join([f"{k}={v}" for k, v in avg_cos.items()])
        L.log(f"    {cos_str}")
    
    return results

def make_hook_fn(storage, key):
    def hook(module, input, output):
        if isinstance(output, tuple):
            storage[key] = output[0].detach().float().cpu()
        else:
            storage[key] = output.detach().float().cpu()
    return hook

# ===================== P364: 注入位置的影响 =====================
def run_p364(model, tokenizer, device, model_name):
    """
    在不同token位置注入方向, 看L0方向对齐度的变化
    
    如果在last_token注入cos≈1但在其他位置注入cos<<1
    → last_token位置特殊(因为它是"空白", 没有强语义)
    """
    L.log("=== P364: 注入位置对L0方向对齐度的影响 ===")
    
    beta = 8.0
    n_layers = len(model.model.layers) if hasattr(model, 'model') and hasattr(model.model, 'layers') else 0
    D = model.config.hidden_size
    
    test_attrs = ["red","sweet","hot","big"]
    
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
        
        noun = "apple"
        prompt = f"The {noun} is"
        toks = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = toks.input_ids
        seq_len = input_ids.shape[1]
        
        L.log(f"    prompt tokens: {input_ids.tolist()}")
        
        position_cos = {}
        
        for pos in range(seq_len):
            embed_layer = model.get_input_embeddings()
            inputs_embeds_base = embed_layer(input_ids).clone()
            
            direction_tensor = torch.tensor(direction, dtype=inputs_embeds_base.dtype, device=device)
            inputs_embeds_intervened = inputs_embeds_base.clone()
            # 在位置pos注入
            inputs_embeds_intervened[0, pos, :] += beta * direction_tensor
            
            # Hook收集第1层输出
            captured_base = {}
            captured_interv = {}
            
            hooks = []
            if n_layers > 0:
                hooks.append(model.model.layers[0].register_forward_hook(
                    make_hook_fn(captured_base, "L0")))
            
            with torch.no_grad():
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                try:
                    _ = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids)
                except:
                    for h in hooks: h.remove()
                    continue
            for h in hooks: h.remove()
            
            hooks2 = []
            if n_layers > 0:
                hooks2.append(model.model.layers[0].register_forward_hook(
                    make_hook_fn(captured_interv, "L0")))
            
            with torch.no_grad():
                try:
                    _ = model(inputs_embeds=inputs_embeds_intervened, position_ids=position_ids)
                except:
                    for h in hooks2: h.remove()
                    continue
            for h in hooks2: h.remove()
            
            if "L0" not in captured_base or "L0" not in captured_interv:
                continue
            
            # 对所有token位置计算方向对齐度
            pos_cos = {}
            for tok_pos in range(seq_len):
                h_base = captured_base["L0"][0, tok_pos, :].numpy()
                h_interv = captured_interv["L0"][0, tok_pos, :].numpy()
                delta = h_interv - h_base
                delta_norm = np.linalg.norm(delta)
                if delta_norm > 1e-8:
                    cos_val = float(np.dot(delta, w_lm_np) / delta_norm)
                else:
                    cos_val = 0.0
                pos_cos[tok_pos] = round(cos_val, 4)
            
            position_cos[f"inject@{pos}"] = pos_cos
        
        results[attr] = position_cos
        
        # 打印
        for inj_pos, cos_dict in position_cos.items():
            cos_vals = [f"L{p}={v}" for p, v in sorted(cos_dict.items())]
            L.log(f"    {inj_pos}: {', '.join(cos_vals)}")
    
    return results

# ===================== 主函数 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    model_name = args.model
    L.log(f"===== Phase LXVIII: L0方向对齐度根源分析 =====")
    L.log(f"模型: {model_name}")
    
    L.log("加载模型...")
    model, tokenizer, device = load_model(model_name)
    n_layers = len(model.model.layers) if hasattr(model, 'model') and hasattr(model.model, 'layers') else 0
    D = model.config.hidden_size
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    
    # 打印关键架构信息
    L.log(f"模型: {model_name}, 层数: {n_layers}, 维度: {D}, 参数量: {n_params:.1f}B")
    
    # 检查架构细节
    if hasattr(model, 'model'):
        if hasattr(model.model, 'embed_norm'):
            L.log(f"  embed_norm: 存在")
        else:
            L.log(f"  embed_norm: 不存在")
        
        if n_layers > 0:
            layer0 = model.model.layers[0]
            L.log(f"  layer0有input_layernorm: {hasattr(layer0, 'input_layernorm')}")
            L.log(f"  layer0有post_attention_layernorm: {hasattr(layer0, 'post_attention_layernorm')}")
            L.log(f"  layer0有self_attn: {hasattr(layer0, 'self_attn')}")
            L.log(f"  layer0有mlp: {hasattr(layer0, 'mlp')}")
            
            if hasattr(layer0, 'self_attn'):
                sa = layer0.self_attn
                L.log(f"  self_attn有q_proj: {hasattr(sa, 'q_proj')}")
                L.log(f"  self_attn有k_proj: {hasattr(sa, 'k_proj')}")
                L.log(f"  self_attn有v_proj: {hasattr(sa, 'v_proj')}")
    
    # P362: RMSNorm影响
    L.log("P362: RMSNorm对注入方向的影响...")
    p362_results = run_p362(model, tokenizer, device, model_name)
    
    # P363: Attention和FFN影响
    L.log("P363: 第1层Attention和FFN对方向的影响...")
    p363_results = run_p363(model, tokenizer, device, model_name)
    
    # P364: 注入位置影响
    L.log("P364: 注入位置对L0方向对齐度的影响...")
    p364_results = run_p364(model, tokenizer, device, model_name)
    
    # 保存结果
    all_results = {
        "model": model_name,
        "n_layers": n_layers,
        "D": D,
        "n_params_B": round(n_params, 2),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "p362_norm_impact": p362_results,
        "p363_attn_ffn_impact": p363_results,
        "p364_position_impact": p364_results,
    }
    
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = OUT_DIR / f"phase_lxviii_p362_364_{model_name}_{ts}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    L.log(f"结果已保存: {out_path}")
    
    # 汇总
    L.log("\n===== 汇总 =====")
    
    L.log("\nP362: 各节点的方向保持度")
    for attr in ALL_ATTRS[:6]:
        if attr in p362_results:
            r = p362_results[attr]
            L.log(f"  {attr}: {r}")
    
    L.log("\nP363: 第1层组件影响")
    for attr in ["red","green","sweet","sour","hot","cold","big","small"]:
        if attr in p363_results:
            r = p363_results[attr]
            L.log(f"  {attr}: {r}")
    
    L.log("\nP364: 注入位置影响")
    for attr in ["red","sweet","hot","big"]:
        if attr in p364_results:
            r = p364_results[attr]
            for inj_pos, cos_dict in r.items():
                L.log(f"  {attr} {inj_pos}: last_token_cos={cos_dict.get(str(len(cos_dict)-1), 'N/A')}")
    
    # 释放GPU
    del model
    gc.collect()
    torch.cuda.empty_cache()
    L.log("GPU已释放")
    L.close()

if __name__ == "__main__":
    main()
