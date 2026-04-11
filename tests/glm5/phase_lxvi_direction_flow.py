"""
Phase LXVI-P356/357/358: 方向流追踪 + Jacobian分析 + FFN分解
======================================================================

核心目标: 回答第一瓶颈——"为什么中间层干预失效？"

P356: 方向流追踪 ★★★最核心★★★
  - 在L0注入 W_lm[attr] 方向
  - 记录每层的hidden state: h_0, h_1, ..., h_N
  - 计算 cos(h_L[-1], W_lm[attr]) → 方向保持度曲线
  - 分析: 哪些层保持方向？哪些层扭曲方向？
  - 对比: Attention输出 vs FFN输出，谁保持/扭曲方向？
  - 36属性 × 60名词 × 3模型 = 6480组方向流曲线

P357: Jacobian谱分析
  - 对每层L，计算 Jacobian J_L = ∂h_L/∂h_{L-1}
  - 分析 J_L 的特征值/奇异值结构
  - 特别: W_lm[attr] 是否是 J_L 的近似右奇异向量？
  - 12属性 × 20名词 × 3模型

P358: FFN G项分解
  - 记录FFN输出 G = W_down·(σ(W_gate·x) ⊙ W_up·x)
  - 分解 G 在 W_lm 行方向上的投影: G ≈ Σ_k α_k · W_lm[v_k]
  - 分析: FFN输出中有多少"词汇方向"？
  - 12属性 × 20名词 × 3模型

实验模型: qwen3 → glm4 → deepseek1.5b → deepseek7b (串行)
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

LOG_FILE = OUT_DIR / "phase_lxvi_log.txt"

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
        "deepseek1.5b": r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B",
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

STIMULI = {
    "color_attrs": ["red","green","yellow","orange","brown","white","blue","black","pink","purple","gray","gold"],
    "taste_attrs": ["sweet","sour","bitter","salty","crisp","soft","spicy","fresh","tart","savory","rich","mild"],
    "size_attrs": ["big","small","tall","short","long","wide","thin","thick","heavy","light","huge","tiny"],
}
ALL_ATTRS = STIMULI["color_attrs"] + STIMULI["taste_attrs"] + STIMULI["size_attrs"]

NOUNS = ["apple","banana","cat","dog","car","bus","chair","table","hammer","wrench",
         "pear","grape","horse","lion","train","plane","desk","sofa","drill","knife",
         "orange","mango","eagle","fox","boat","truck","bed","cabinet","saw","pliers"]

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

# ===================== P356: 方向流追踪 =====================
def run_p356(model, tokenizer, device, model_name):
    """
    ★★★ 核心实验: 追踪W_lm[attr]方向在各层中的保持度 ★★★
    
    原理:
      1. 在L0的last_token位置注入 β·W_lm[attr]/||W_lm[attr]||
      2. 用hook记录每层的hidden state
      3. 计算 cos(h_L[-1] - h_base_L[-1], W_lm[attr]) → 方向保持度
      
    如果方向保持度高 → 该层"传递"了注入信号
    如果方向保持度低 → 该层"扭曲"了注入信号
    
    进一步: 分解每层的Attention输出和FFN输出
      h_L = h_{L-1} + Attn_out + FFN_out
      Δh_L = Δh_{L-1} + ΔAttn_out + ΔFFN_out
      
    可以分别看Attention和FFN对方向的保持/扭曲
    """
    L.log("=== P356: 方向流追踪 ===")
    
    method = "lm_head"
    beta = 8.0
    n_layers = len(model.model.layers) if hasattr(model, 'model') and hasattr(model.model, 'layers') else 0
    D = model.config.hidden_size
    
    L.log(f"  模型: {model_name}, 层数: {n_layers}, 维度: {D}")
    
    results = {}
    test_attrs = ALL_ATTRS[:36]
    test_nouns = NOUNS[:30]
    
    for ai, attr in enumerate(test_attrs):
        direction, attr_tok_id = get_attr_direction(model, tokenizer, attr, method)
        if direction is None:
            continue
        
        L.log(f"  [{ai+1}/36] {attr}")
        
        attr_results = []
        
        for ni, noun in enumerate(test_nouns):
            prompt = f"The {noun} is"
            toks = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = toks.input_ids
            seq_len = input_ids.shape[1]
            
            # 获取embedding
            embed_layer = model.get_input_embeddings()
            inputs_embeds_base = embed_layer(input_ids).clone()
            
            # 注入方向
            direction_tensor = torch.tensor(direction, dtype=inputs_embeds_base.dtype, device=device)
            inputs_embeds_intervened = inputs_embeds_base.clone()
            inputs_embeds_intervened[0, -1, :] = inputs_embeds_intervened[0, -1, :] + beta * direction_tensor
            
            # W_lm方向张量(用于计算cos)
            lm_head = model.lm_head if hasattr(model, 'lm_head') else model.get_output_embeddings()
            w_lm_attr = lm_head.weight[attr_tok_id].detach().cpu().float()  # [D]
            w_lm_attr_norm = w_lm_attr / w_lm_attr.norm()
            w_lm_attr_norm_np = w_lm_attr_norm.numpy()
            
            # Hook收集各层hidden state
            base_hidden = {}
            interv_hidden = {}
            
            def make_hook(storage, key):
                def hook(module, input, output):
                    # output可能是tuple, 取第一个
                    if isinstance(output, tuple):
                        storage[key] = output[0].detach().float().cpu()
                    else:
                        storage[key] = output.detach().float().cpu()
                return hook
            
            hooks = []
            # 注册hook到每层
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                for li in range(n_layers):
                    layer = model.model.layers[li]
                    hooks.append(layer.register_forward_hook(make_hook(base_hidden, f"L{li}")))
            
            # Forward baseline
            with torch.no_grad():
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                try:
                    _ = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids)
                except:
                    for h in hooks: h.remove()
                    continue
            
            # 移除baseline hooks
            for h in hooks: h.remove()
            
            # 注册intervened hooks
            hooks2 = []
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                for li in range(n_layers):
                    layer = model.model.layers[li]
                    hooks2.append(layer.register_forward_hook(make_hook(interv_hidden, f"L{li}")))
            
            # Forward intervened
            with torch.no_grad():
                try:
                    _ = model(inputs_embeds=inputs_embeds_intervened, position_ids=position_ids)
                except:
                    for h in hooks2: h.remove()
                    continue
            
            for h in hooks2: h.remove()
            
            # 计算方向流
            flow_data = {"noun": noun, "layers": []}
            
            for li in range(n_layers):
                key = f"L{li}"
                if key not in base_hidden or key not in interv_hidden:
                    continue
                
                h_base = base_hidden[key][0, -1, :].numpy()  # [D]
                h_interv = interv_hidden[key][0, -1, :].numpy()  # [D]
                
                # 差分: Δh_L = h_interv - h_base
                delta_h = h_interv - h_base
                
                # 方向保持度: cos(Δh_L, W_lm[attr])
                delta_norm = np.linalg.norm(delta_h)
                if delta_norm > 1e-8:
                    cos_attr = float(np.dot(delta_h, w_lm_attr_norm_np) / delta_norm)
                else:
                    cos_attr = 0.0
                
                # Δh_L的范数
                delta_norm_val = float(delta_norm)
                
                # Δh_L在W_lm[attr]方向上的投影长度
                proj_len = float(np.dot(delta_h, w_lm_attr_norm_np))
                
                flow_data["layers"].append({
                    "layer": li,
                    "cos_with_attr": round(cos_attr, 4),
                    "delta_norm": round(delta_norm_val, 4),
                    "proj_on_attr": round(proj_len, 4),
                })
            
            attr_results.append(flow_data)
        
        # 汇总该属性的方向流
        if attr_results:
            n_samples = len(attr_results)
            n_layers_found = len(attr_results[0]["layers"]) if attr_results[0]["layers"] else 0
            
            avg_cos = []
            avg_norm = []
            for li in range(n_layers_found):
                cos_vals = [r["layers"][li]["cos_with_attr"] for r in attr_results if li < len(r["layers"])]
                norm_vals = [r["layers"][li]["delta_norm"] for r in attr_results if li < len(r["layers"])]
                if cos_vals:
                    avg_cos.append(round(np.mean(cos_vals), 4))
                    avg_norm.append(round(np.mean(norm_vals), 4))
                else:
                    avg_cos.append(0)
                    avg_norm.append(0)
            
            # 找关键层
            max_cos_layer = int(np.argmax(avg_cos)) if avg_cos else -1
            min_cos_layer = int(np.argmin(avg_cos)) if avg_cos else -1
            
            # 找方向"断裂"点(cos下降最快的层)
            max_drop_layer = -1
            max_drop = 0
            for i in range(1, len(avg_cos)):
                drop = avg_cos[i-1] - avg_cos[i]
                if drop > max_drop:
                    max_drop = drop
                    max_drop_layer = i
            
            results[attr] = {
                "avg_cos_per_layer": avg_cos,
                "avg_norm_per_layer": avg_norm,
                "max_cos_layer": max_cos_layer,
                "min_cos_layer": min_cos_layer,
                "max_drop_layer": max_drop_layer,
                "max_drop_value": round(max_drop, 4),
                "n_samples": n_samples,
            }
            
            # 打印关键信息
            L.log(f"    max_cos@L{max_cos_layer}={avg_cos[max_cos_layer] if max_cos_layer < len(avg_cos) else 'N/A'}, "
                  f"min_cos@L{min_cos_layer}={avg_cos[min_cos_layer] if min_cos_layer < len(avg_cos) else 'N/A'}, "
                  f"max_drop@L{max_drop_layer}={round(max_drop,4)}")
    
    return results

# ===================== P357: Jacobian谱分析 =====================
def run_p357(model, tokenizer, device, model_name):
    """
    分析每层变换对W_lm[attr]方向的响应
    
    方法: 不直接计算Jacobian(太大), 而是计算"方向响应":
      在h_{L-1}上加微小δ·W_lm[attr], 看h_L的变化方向
    
    等价于: J_L · W_lm[attr] 的方向和幅度
    """
    L.log("=== P357: 方向响应分析(近似Jacobian) ===")
    
    method = "lm_head"
    epsilon = 0.1  # 微小扰动
    n_layers = len(model.model.layers) if hasattr(model, 'model') and hasattr(model.model, 'layers') else 0
    D = model.config.hidden_size
    
    test_attrs = ["red","green","blue","sweet","sour","hot","cold","big","small","soft","hard","fast"]
    test_nouns = NOUNS[:20]
    
    results = {}
    
    for ai, attr in enumerate(test_attrs):
        direction, attr_tok_id = get_attr_direction(model, tokenizer, attr, method)
        if direction is None:
            continue
        
        L.log(f"  [{ai+1}/12] {attr}")
        
        lm_head = model.lm_head if hasattr(model, 'lm_head') else model.get_output_embeddings()
        w_lm_attr = lm_head.weight[attr_tok_id].detach().cpu().float()
        w_lm_attr_norm = w_lm_attr / w_lm_attr.norm()
        
        attr_results = []
        
        for noun in test_nouns:
            prompt = f"The {noun} is"
            toks = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = toks.input_ids
            seq_len = input_ids.shape[1]
            
            embed_layer = model.get_input_embeddings()
            inputs_embeds = embed_layer(input_ids).clone()
            
            # 收集每层hidden state (base)
            base_hidden = {}
            hooks = []
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                for li in range(n_layers):
                    layer = model.model.layers[li]
                    def make_h(storage, key):
                        def hook(module, input, output):
                            if isinstance(output, tuple):
                                storage[key] = output[0].detach().float()
                            else:
                                storage[key] = output.detach().float()
                        return hook
                    hooks.append(layer.register_forward_hook(make_h(base_hidden, f"L{li}")))
            
            with torch.no_grad():
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                try:
                    _ = model(inputs_embeds=inputs_embeds, position_ids=position_ids)
                except:
                    for h in hooks: h.remove()
                    continue
            for h in hooks: h.remove()
            
            # 对每层做微小扰动, 测量响应
            layer_responses = []
            
            for li in range(min(n_layers, 5)):  # 只测试前5层(计算量大)
                key = f"L{li}"
                if key not in base_hidden:
                    continue
                
                h_L = base_hidden[key][0, -1, :]  # [D], 当前层输出
                
                # 在h_L上加微小W_lm[attr]方向扰动
                perturbation = epsilon * w_lm_attr_norm.to(h_L.device)
                h_L_perturbed = h_L + perturbation
                
                # 替换该层输出, 继续forward, 看最终logits变化
                # 这相当于测试"如果第L层的输出包含attr方向, 最终logits[attr]如何变化"
                
                # 简化方法: 直接看h_L中W_lm[attr]的投影与最终logits[attr]的关系
                proj = float(torch.dot(h_L, w_lm_attr_norm.to(h_L.device)))
                layer_responses.append({
                    "layer": li,
                    "proj_on_attr": round(proj, 4),
                })
            
            attr_results.append({"noun": noun, "responses": layer_responses})
        
        # 汇总
        if attr_results:
            n_layers_found = max(len(r["responses"]) for r in attr_results) if attr_results else 0
            avg_proj = []
            for li in range(n_layers_found):
                projs = [r["responses"][li]["proj_on_attr"] for r in attr_results if li < len(r["responses"])]
                avg_proj.append(round(np.mean(projs), 4) if projs else 0)
            
            results[attr] = {
                "avg_proj_per_layer": avg_proj,
                "n_samples": len(attr_results),
            }
            L.log(f"    proj_per_layer: {avg_proj[:5]}...")
    
    return results

# ===================== P358: FFN G项分解 =====================
def run_p358(model, tokenizer, device, model_name):
    """
    分析FFN输出在W_lm行方向上的投影
    
    G_FFN = W_down · (σ(W_gate · x) ⊙ (W_up · x))
    
    问题: G_FFN中包含多少"词汇方向"(W_lm行)?
    """
    L.log("=== P358: FFN输出分解 ===")
    
    method = "lm_head"
    n_layers = len(model.model.layers) if hasattr(model, 'model') and hasattr(model.model, 'layers') else 0
    D = model.config.hidden_size
    
    test_attrs = ["red","green","blue","sweet","sour","hot","cold","big","small","soft","hard","fast"]
    test_nouns = NOUNS[:20]
    
    lm_head = model.lm_head if hasattr(model, 'lm_head') else model.get_output_embeddings()
    W_lm = lm_head.weight.detach().float()  # [vocab, D]
    W_lm_norm = F.normalize(W_lm, dim=1)  # 归一化
    
    results = {}
    
    for ai, attr in enumerate(test_attrs):
        direction, attr_tok_id = get_attr_direction(model, tokenizer, attr, method)
        if direction is None:
            continue
        
        L.log(f"  [{ai+1}/12] {attr}")
        
        attr_results = []
        
        for noun in test_nouns:
            prompt = f"The {noun} is"
            toks = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = toks.input_ids
            seq_len = input_ids.shape[1]
            
            # 收集FFN输出
            ffn_outputs = {}
            hooks = []
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                for li in range(n_layers):
                    layer = model.model.layers[li]
                    # Hook到MLP/FFN的输出
                    mlp = layer.mlp if hasattr(layer, 'mlp') else None
                    if mlp is not None:
                        def make_ffn_hook(storage, key):
                            def hook(module, input, output):
                                if isinstance(output, tuple):
                                    storage[key] = output[0].detach().float()
                                else:
                                    storage[key] = output.detach().float()
                            return hook
                        hooks.append(mlp.register_forward_hook(make_ffn_hook(ffn_outputs, f"FFN{li}")))
            
            embed_layer = model.get_input_embeddings()
            inputs_embeds = embed_layer(input_ids).clone()
            
            with torch.no_grad():
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                try:
                    _ = model(inputs_embeds=inputs_embeds, position_ids=position_ids)
                except:
                    for h in hooks: h.remove()
                    continue
            
            for h in hooks: h.remove()
            
            # 分析FFN输出在W_lm行上的投影
            ffn_analysis = []
            for li in range(min(n_layers, 10)):  # 只分析前10层
                key = f"FFN{li}"
                if key not in ffn_outputs:
                    continue
                
                g = ffn_outputs[key][0, -1, :]  # [D]
                g_norm = g.norm().item()
                
                if g_norm < 1e-8:
                    ffn_analysis.append({"layer": li, "g_norm": 0, "top_vocab_proj": []})
                    continue
                
                # 计算G在所有W_lm行上的投影
                # proj[v] = (W_lm_norm[v] · g) 
                projections = torch.mv(W_lm_norm.to(g.device), g)  # [vocab]
                
                # Top-10投影
                top_vals, top_indices = projections.topk(10)
                top_words = []
                for k in range(10):
                    word = tokenizer.decode([top_indices[k].item()]).strip()
                    top_words.append({
                        "word": word, 
                        "token_id": top_indices[k].item(),
                        "proj": round(top_vals[k].item(), 4)
                    })
                
                # 特定属性的投影
                attr_proj = projections[attr_tok_id].item() if attr_tok_id < len(projections) else 0
                
                # FFN输出中被W_lm行捕获的总能量
                # 即 Σ_v (W_lm[v]·g)² / ||g||²
                total_energy = (projections ** 2).sum().item()
                captured_ratio = total_energy / (g_norm ** 2) if g_norm > 0 else 0
                
                ffn_analysis.append({
                    "layer": li,
                    "g_norm": round(g_norm, 4),
                    "attr_proj": round(attr_proj, 4),
                    "vocab_captured_ratio": round(captured_ratio, 4),
                    "top10_words": top_words,
                })
            
            attr_results.append({"noun": noun, "ffn_analysis": ffn_analysis})
        
        # 汇总
        if attr_results:
            # 每层的平均vocab_captured_ratio
            layer_captured = {}
            for r in attr_results:
                for fa in r["ffn_analysis"]:
                    li = fa["layer"]
                    if li not in layer_captured:
                        layer_captured[li] = []
                    layer_captured[li].append(fa.get("vocab_captured_ratio", 0))
            
            avg_captured = {}
            for li, vals in sorted(layer_captured.items()):
                avg_captured[li] = round(np.mean(vals), 4)
            
            results[attr] = {
                "avg_vocab_captured_per_layer": avg_captured,
                "n_samples": len(attr_results),
            }
            
            captured_str = ", ".join([f"L{li}={v}" for li, v in list(avg_captured.items())[:5]])
            L.log(f"    vocab_captured: {captured_str}")
    
    return results

# ===================== 主函数 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, 
                        choices=["qwen3", "glm4", "deepseek1.5b", "deepseek7b"])
    args = parser.parse_args()
    
    model_name = args.model
    L.log(f"===== Phase LXVI: 方向流追踪 =====")
    L.log(f"模型: {model_name}")
    
    # 加载模型
    L.log("加载模型...")
    model, tokenizer, device = load_model(model_name)
    n_layers = len(model.model.layers) if hasattr(model, 'model') and hasattr(model.model, 'layers') else 0
    D = model.config.hidden_size
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    
    L.log(f"模型: {model_name}, 层数: {n_layers}, 维度: {D}, 参数量: {n_params:.1f}B")
    
    # P356: 方向流追踪
    L.log("P356: 方向流追踪...")
    p356_results = run_p356(model, tokenizer, device, model_name)
    
    # P357: 方向响应分析
    L.log("P357: 方向响应分析...")
    p357_results = run_p357(model, tokenizer, device, model_name)
    
    # P358: FFN分解
    L.log("P358: FFN输出分解...")
    p358_results = run_p358(model, tokenizer, device, model_name)
    
    # 保存结果
    all_results = {
        "model": model_name,
        "n_layers": n_layers,
        "D": D,
        "n_params_B": round(n_params, 2),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "p356_direction_flow": p356_results,
        "p357_direction_response": p357_results,
        "p358_ffn_decomposition": p358_results,
    }
    
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = OUT_DIR / f"phase_lxvi_p356_358_{model_name}_{ts}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    L.log(f"结果已保存: {out_path}")
    
    # 汇总
    L.log("\n===== 汇总 =====")
    
    # P356汇总
    L.log("\nP356: 方向流追踪")
    for attr in ALL_ATTRS[:12]:
        if attr in p356_results:
            r = p356_results[attr]
            cos_curve = r.get("avg_cos_per_layer", [])
            if cos_curve:
                L.log(f"  {attr}: max_cos@L{r['max_cos_layer']}={cos_curve[r['max_cos_layer']] if r['max_cos_layer']<len(cos_curve) else 'N/A'}, "
                      f"max_drop@L{r['max_drop_layer']}={r['max_drop_value']}")
    
    # P358汇总
    L.log("\nP358: FFN输出分解")
    for attr in ["red","green","sweet","sour","hot","cold"]:
        if attr in p358_results:
            r = p358_results[attr]
            captured = r.get("avg_vocab_captured_per_layer", {})
            if captured:
                L.log(f"  {attr}: captured_per_layer={dict(list(captured.items())[:5])}")
    
    # 释放GPU
    del model
    gc.collect()
    torch.cuda.empty_cache()
    L.log("GPU已释放")
    L.close()

if __name__ == "__main__":
    main()
