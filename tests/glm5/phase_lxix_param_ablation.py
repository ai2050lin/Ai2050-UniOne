"""
Phase LXIX-P365/366/367: 参数消元与"强制平直"验证
======================================================================

验证Phase LXVIII的假说: GLM4的第1层是"方向恢复器"

P365: RMSNorm权重γ消元
  - 条件1: 原始γ (基线)
  - 条件2: γ=全向量均值 (抹去γ的"形状"但保持缩放) 
  - 条件3: γ=1 (去掉缩放, 只保留归一化) - 注意: 某些模型gamma很小, γ=1会爆炸
  - 条件4: γ=abs(γ).sign()*mean(γ) (保持符号但抹去形状)
  关键: 不直接用γ=1, 而是保持γ的范数不变, 只改变"方向"

P366: Attn/FFN屏蔽实验
  - 正常: Attn和FFN都正常
  - Attn=0: 将L0的Attn输出置0
  - FFN=0: 将L0的FFN输出置0
  - Both=0: Attn=0+FFN=0 (只保留RMSNorm+残差)

P367: β梯度扫描
  - β从0.01到20, 寻找cos开始崩溃的临界点

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
LOG_FILE = OUT_DIR / "phase_lxix_log.txt"

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

def compute_intervention_cos_v2(model, tokenizer, device, attr, beta, 
                                 input_ids, seq_len, inputs_embeds_base,
                                 direction, w_lm_np,
                                 attn_zero=False, ffn_zero=False):
    """
    通用干预函数: 注入方向, 收集L0输出的cos
    """
    direction_tensor = torch.tensor(direction, dtype=inputs_embeds_base.dtype, device=device)
    inputs_embeds_intervened = inputs_embeds_base.clone()
    inputs_embeds_intervened[0, -1, :] += beta * direction_tensor
    
    n_layers = len(model.model.layers) if hasattr(model, 'model') and hasattr(model.model, 'layers') else 0
    if n_layers == 0:
        return None
    
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    
    # === Base forward ===
    captured_base = {}
    hooks_base = []
    hooks_base.append(model.model.layers[0].register_forward_hook(
        make_hook_fn(captured_base, "L0")))
    
    with torch.no_grad():
        try:
            _ = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids)
        except Exception as e:
            for h in hooks_base: h.remove()
            return None
    for h in hooks_base: h.remove()
    
    # === Intervened forward ===
    captured_interv = {}
    hooks_interv = []
    hooks_interv.append(model.model.layers[0].register_forward_hook(
        make_hook_fn(captured_interv, "L0")))
    
    # 屏蔽hook
    zero_hooks = []
    
    if attn_zero:
        layer0 = model.model.layers[0]
        if hasattr(layer0, 'self_attn'):
            def zero_attn_hook(module, input, output):
                if isinstance(output, tuple):
                    return (torch.zeros_like(output[0]),) + output[1:]
                else:
                    return torch.zeros_like(output)
            zero_hooks.append(layer0.self_attn.register_forward_hook(zero_attn_hook))
    
    if ffn_zero:
        layer0 = model.model.layers[0]
        if hasattr(layer0, 'mlp'):
            def zero_ffn_hook(module, input, output):
                if isinstance(output, tuple):
                    return (torch.zeros_like(output[0]),) + output[1:]
                else:
                    return torch.zeros_like(output)
            zero_hooks.append(layer0.mlp.register_forward_hook(zero_ffn_hook))
    
    with torch.no_grad():
        try:
            _ = model(inputs_embeds=inputs_embeds_intervened, position_ids=position_ids)
        except Exception as e:
            for h in hooks_interv: h.remove()
            for h in zero_hooks: h.remove()
            return None
    for h in hooks_interv: h.remove()
    for h in zero_hooks: h.remove()
    
    # 计算cos
    if "L0" not in captured_base or "L0" not in captured_interv:
        return None
    
    h_base = captured_base["L0"][0, -1, :].numpy()
    h_interv = captured_interv["L0"][0, -1, :].numpy()
    delta_h = h_interv - h_base
    delta_norm = np.linalg.norm(delta_h)
    
    if delta_norm > 1e-8 and not np.isnan(delta_norm) and not np.isinf(delta_norm):
        cos_val = float(np.dot(delta_h, w_lm_np) / delta_norm)
        if np.isnan(cos_val) or np.isinf(cos_val):
            cos_val = 0.0
    else:
        cos_val = 0.0
    
    return cos_val


# ===================== P365: RMSNorm权重γ消元 =====================
def run_p365(model, tokenizer, device, model_name):
    """
    ★★★ 核心实验: RMSNorm的γ是否是方向恢复的关键? ★★★
    
    关键修正: γ=1会爆炸(因为Qwen3的γ均值仅0.023!)
    
    改用两种消元方式:
    1. γ=均值: 抹去γ的"形状"(各维度差异), 保留缩放
    2. γ归一化: 将γ归一化到||γ||=原始||γ||但方向均匀(随机方向×范数)
    """
    L.log("=== P365: RMSNorm权重γ消元 ===")
    
    beta = 8.0
    n_layers = len(model.model.layers) if hasattr(model, 'model') and hasattr(model.model, 'layers') else 0
    
    test_attrs = ["red", "sweet", "hot", "big"]
    results = {}
    
    if n_layers == 0:
        L.log("  无Transformer层, 跳过")
        return results
    
    layer0 = model.model.layers[0]
    if not hasattr(layer0, 'input_layernorm'):
        L.log("  无input_layernorm, 跳过")
        return results
    
    norm_layer = layer0.input_layernorm
    if not hasattr(norm_layer, 'weight'):
        L.log("  无weight参数, 跳过")
        return results
    
    # 保存原始γ
    original_gamma = norm_layer.weight.data.clone()
    gamma_mean = original_gamma.mean().item()
    gamma_std = original_gamma.std().item()
    gamma_norm = original_gamma.norm().item()
    
    L.log(f"  原始γ: mean={gamma_mean:.4f}, std={gamma_std:.4f}, "
          f"norm={gamma_norm:.4f}, min={original_gamma.min().item():.4f}, max={original_gamma.max().item():.4f}")
    
    for ai, attr in enumerate(test_attrs):
        direction, attr_tok_id = get_attr_direction(model, tokenizer, attr, "lm_head")
        if direction is None:
            continue
        
        L.log(f"  [{ai+1}/4] {attr}")
        
        lm_head = model.lm_head if hasattr(model, 'lm_head') else model.get_output_embeddings()
        w_lm_attr = lm_head.weight[attr_tok_id].detach().cpu().float()
        w_lm_attr_norm = w_lm_attr / w_lm_attr.norm()
        w_lm_np = w_lm_attr_norm.numpy()
        
        attr_cos = {}
        
        for noun in NOUNS[:3]:
            prompt = f"The {noun} is"
            toks = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = toks.input_ids
            seq_len = input_ids.shape[1]
            embed_layer = model.get_input_embeddings()
            inputs_embeds_base = embed_layer(input_ids).clone()
            
            noun_cos = {}
            
            # 条件1: 原始γ
            norm_layer.weight.data.copy_(original_gamma)
            cos_orig = compute_intervention_cos_v2(
                model, tokenizer, device, attr, beta,
                input_ids, seq_len, inputs_embeds_base,
                direction, w_lm_np)
            noun_cos["original"] = round(cos_orig, 4) if cos_orig is not None else None
            
            # 条件2: γ=均值(抹去形状, 保留缩放)
            norm_layer.weight.data.fill_(gamma_mean)
            cos_mean = compute_intervention_cos_v2(
                model, tokenizer, device, attr, beta,
                input_ids, seq_len, inputs_embeds_base,
                direction, w_lm_np)
            noun_cos["gamma_mean"] = round(cos_mean, 4) if cos_mean is not None else None
            
            # 条件3: γ=随机方向×原始范数 (保持范数但随机方向)
            torch.manual_seed(42)
            rand_dir = torch.randn_like(original_gamma)
            rand_dir = rand_dir / rand_dir.norm() * gamma_norm
            norm_layer.weight.data.copy_(rand_dir)
            cos_rand = compute_intervention_cos_v2(
                model, tokenizer, device, attr, beta,
                input_ids, seq_len, inputs_embeds_base,
                direction, w_lm_np)
            noun_cos["gamma_rand"] = round(cos_rand, 4) if cos_rand is not None else None
            
            # 条件4: γ=1.0 (极端: 标准RMSNorm缩放)
            # 注意: 如果原始γ很小, γ=1会爆炸
            norm_layer.weight.data.fill_(1.0)
            cos_g1 = compute_intervention_cos_v2(
                model, tokenizer, device, attr, beta,
                input_ids, seq_len, inputs_embeds_base,
                direction, w_lm_np)
            noun_cos["gamma_1"] = round(cos_g1, 4) if cos_g1 is not None else None
            
            # 恢复原始γ
            norm_layer.weight.data.copy_(original_gamma)
            
            attr_cos[noun] = noun_cos
        
        results[attr] = attr_cos
        for noun, vals in attr_cos.items():
            L.log(f"    {noun}: orig={vals['original']}, γ_mean={vals['gamma_mean']}, "
                  f"γ_rand={vals['gamma_rand']}, γ=1={vals['gamma_1']}")
        
        # 汇总
        avg = {}
        for cond in ["original", "gamma_mean", "gamma_rand", "gamma_1"]:
            vals = [v[cond] for v in attr_cos.values() if v[cond] is not None]
            if vals:
                avg[cond] = round(np.mean(vals), 4)
        L.log(f"    平均: {avg}")
    
    # 确保恢复原始γ
    norm_layer.weight.data.copy_(original_gamma)
    
    return results


# ===================== P366: Attn/FFN屏蔽实验 =====================
def run_p366(model, tokenizer, device, model_name):
    """
    ★★★ 量化Attn和FFN的"污染分量" ★★★
    
    四种条件:
    1. 正常: Attn和FFN都正常
    2. Attn=0: 将L0的Attn输出置0 → 只保留残差+FFN
    3. FFN=0: 将L0的FFN输出置0 → 只保留残差+Attn
    4. Both=0: 只保留残差连接 → 纯"直通"信号
    
    h_L0 = input_layernorm(h) + Attn_out + FFN_out
    (实际: h_L0 = h + Attn(input_layernorm(h)) + FFN(post_attn_layernorm(h + Attn_out)))
    """
    L.log("=== P366: Attn/FFN屏蔽实验 ===")
    
    beta = 8.0
    n_layers = len(model.model.layers) if hasattr(model, 'model') and hasattr(model.model, 'layers') else 0
    
    if n_layers == 0:
        L.log("  无Transformer层, 跳过")
        return {}
    
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
        
        attr_cos = {}
        
        for noun in NOUNS[:3]:
            prompt = f"The {noun} is"
            toks = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = toks.input_ids
            seq_len = input_ids.shape[1]
            embed_layer = model.get_input_embeddings()
            inputs_embeds_base = embed_layer(input_ids).clone()
            
            noun_cos = {}
            
            # 条件1: 正常
            cos_normal = compute_intervention_cos_v2(
                model, tokenizer, device, attr, beta,
                input_ids, seq_len, inputs_embeds_base,
                direction, w_lm_np)
            noun_cos["normal"] = round(cos_normal, 4) if cos_normal is not None else None
            
            # 条件2: Attn=0
            cos_attn0 = compute_intervention_cos_v2(
                model, tokenizer, device, attr, beta,
                input_ids, seq_len, inputs_embeds_base,
                direction, w_lm_np, attn_zero=True)
            noun_cos["attn0"] = round(cos_attn0, 4) if cos_attn0 is not None else None
            
            # 条件3: FFN=0
            cos_ffn0 = compute_intervention_cos_v2(
                model, tokenizer, device, attr, beta,
                input_ids, seq_len, inputs_embeds_base,
                direction, w_lm_np, ffn_zero=True)
            noun_cos["ffn0"] = round(cos_ffn0, 4) if cos_ffn0 is not None else None
            
            # 条件4: Attn=0+FFN=0
            cos_both0 = compute_intervention_cos_v2(
                model, tokenizer, device, attr, beta,
                input_ids, seq_len, inputs_embeds_base,
                direction, w_lm_np, attn_zero=True, ffn_zero=True)
            noun_cos["both0"] = round(cos_both0, 4) if cos_both0 is not None else None
            
            attr_cos[noun] = noun_cos
        
        results[attr] = attr_cos
        for noun, vals in attr_cos.items():
            L.log(f"    {noun}: normal={vals['normal']}, attn0={vals['attn0']}, "
                  f"ffn0={vals['ffn0']}, both0={vals['both0']}")
        
        # 汇总
        avg = {}
        for cond in ["normal", "attn0", "ffn0", "both0"]:
            vals = [v[cond] for v in attr_cos.values() if v[cond] is not None]
            if vals:
                avg[cond] = round(np.mean(vals), 4)
        L.log(f"    平均: {avg}")
    
    return results


# ===================== P367: β梯度扫描 =====================
def run_p367(model, tokenizer, device, model_name):
    """
    ★★★ 寻找GLM4的"线性边界" ★★★
    
    β从0.01到20, 逐步扫描
    记录每个β对应的cos(Δh_L0, W_lm[attr])
    """
    L.log("=== P367: β梯度扫描 - 寻找线性边界 ===")
    
    n_layers = len(model.model.layers) if hasattr(model, 'model') and hasattr(model.model, 'layers') else 0
    
    betas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0]
    
    test_attrs = ["red", "sweet"]
    results = {}
    
    if n_layers == 0:
        L.log("  无Transformer层, 跳过")
        return results
    
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
        
        beta_cos = {}
        
        for beta in betas:
            cos_val = compute_intervention_cos_v2(
                model, tokenizer, device, attr, beta,
                input_ids, seq_len, inputs_embeds_base,
                direction, w_lm_np)
            
            beta_cos[str(beta)] = round(cos_val, 4) if cos_val is not None else None
            L.log(f"    β={beta}: cos={beta_cos[str(beta)]}")
        
        results[attr] = beta_cos
        
        # 找到线性边界
        cos_vals = [(b, beta_cos[str(b)]) for b in betas if beta_cos[str(b)] is not None]
        if len(cos_vals) > 1:
            max_cos = max(v for _, v in cos_vals)
            if max_cos > 0:
                threshold = 0.9 * max_cos
                boundary = None
                for i in range(1, len(cos_vals)):
                    if cos_vals[i][1] < threshold and cos_vals[i-1][1] >= threshold:
                        boundary = cos_vals[i][0]
                        break
                L.log(f"    线性边界(90%阈值): β≈{boundary}")
                results[attr]["_linear_boundary"] = boundary
    
    return results


# ===================== 主函数 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--experiment", type=str, default="all",
                        choices=["all", "p365", "p366", "p367"])
    args = parser.parse_args()
    
    model_name = args.model
    L.log(f"===== Phase LXIX: 参数消元与强制平直验证 =====")
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
    
    if args.experiment in ["all", "p365"]:
        L.log("P365: RMSNorm权重γ消元...")
        all_results["p365"] = run_p365(model, tokenizer, device, model_name)
    
    if args.experiment in ["all", "p366"]:
        L.log("P366: Attn/FFN屏蔽实验...")
        all_results["p366"] = run_p366(model, tokenizer, device, model_name)
    
    if args.experiment in ["all", "p367"]:
        L.log("P367: β梯度扫描...")
        all_results["p367"] = run_p367(model, tokenizer, device, model_name)
    
    # 保存结果
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = OUT_DIR / f"phase_lxix_p365_367_{model_name}_{ts}.json"
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
