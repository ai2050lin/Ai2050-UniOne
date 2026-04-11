"""
Phase LXVII-P359/360/361: W_lm与W_embed对齐度验证 + 层变换信号保真度 + 多方向正交性
======================================================================

核心目标: 验证Phase LXVI的核心假设——"W_lm与W_embed的对齐度决定干预成败"

P359: ★★★最核心★★★ W_lm与W_embed对齐度
  - 对每个属性词, 计算 cos(W_lm[attr], W_embed[attr])
  - 对比四模型的对齐度
  - 如果GLM4的cos≈1而其他模型cos<<1, 则证实假设

P360: 层变换信号保真度
  - 对每层, 计算: 在h_{L-1}上加β·W_lm[attr], 经该层后h_L中的方向保持度
  - 即逐层计算 cos(Δh_L, W_lm[attr]) / cos(Δh_{L-1}, W_lm[attr])
  - 这是"每层的方向保持率"

P361: 多方向正交性与组合干预
  - 计算W_lm[attr1]和W_lm[attr2]之间的cos
  - 寻找正交属性对(cos≈0)
  - 测试正交属性对的同时干预效果

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
LOG_FILE = OUT_DIR / "phase_lxvii_log.txt"

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

STIMULI = {
    "color_attrs": ["red","green","yellow","orange","brown","white","blue","black","pink","purple","gray","gold"],
    "taste_attrs": ["sweet","sour","bitter","salty","crisp","soft","spicy","fresh","tart","savory","rich","mild"],
    "size_attrs": ["big","small","tall","short","long","wide","thin","thick","heavy","light","huge","tiny"],
}
ALL_ATTRS = STIMULI["color_attrs"] + STIMULI["taste_attrs"] + STIMULI["size_attrs"]

NOUNS = ["apple","banana","cat","dog","car","bus","chair","table","hammer","wrench",
         "pear","grape","horse","lion","train","plane","desk","sofa","drill","knife"]

# ===================== P359: W_lm与W_embed对齐度 =====================
def run_p359(model, tokenizer, device, model_name):
    """
    ★★★ 核心验证: cos(W_lm[attr], W_embed[attr]) ★★★
    
    假设: GLM4的cos≈1.0, 其他模型cos<<1.0
    
    如果假设成立 → 解释了GLM4为何是唯一"全面有效"的模型
    如果假设不成立 → 需要寻找其他原因
    """
    L.log("=== P359: W_lm与W_embed对齐度验证 ===")
    
    lm_head = model.lm_head if hasattr(model, 'lm_head') else model.get_output_embeddings()
    embed_layer = model.get_input_embeddings()
    
    # 检查tied: 不加载全矩阵, 而是采样比较
    vocab_size = lm_head.weight.shape[0]
    D = lm_head.weight.shape[1]
    L.log(f"  W_lm shape: [{vocab_size}, {D}]")
    
    # 采样1000个token检查tied
    sample_size = min(1000, vocab_size)
    sample_ids = np.random.choice(vocab_size, sample_size, replace=False)
    
    is_tied = True
    cos_values = []
    for vid in sample_ids:
        w_lm_v = lm_head.weight[vid].detach().cpu().float()
        w_emb_v = embed_layer.weight[vid].detach().cpu().float()
        if not torch.allclose(w_lm_v, w_emb_v, atol=1e-4):
            is_tied = False
        cos_v = F.cosine_similarity(w_lm_v.unsqueeze(0), w_emb_v.unsqueeze(0)).item()
        cos_values.append(cos_v)
    
    L.log(f"  W_lm == W_embed (tied)? {is_tied}")
    
    avg_cos = np.mean(cos_values)
    median_cos = np.median(cos_values)
    L.log(f"  全局对齐度: avg_cos={avg_cos:.4f}, median_cos={median_cos:.4f}")
    
    # 对所有属性词计算对齐度
    attr_alignment = {}
    for attr in ALL_ATTRS:
        attr_tok_ids = tokenizer.encode(attr, add_special_tokens=False)
        if len(attr_tok_ids) == 0:
            continue
        
        tok_id = attr_tok_ids[0]
        if tok_id >= vocab_size:
            continue
        
        w_lm_attr = lm_head.weight[tok_id].detach().cpu().float()
        w_emb_attr = embed_layer.weight[tok_id].detach().cpu().float()
        
        # cos(W_lm[attr], W_embed[attr])
        cos_attr = F.cosine_similarity(w_lm_attr.unsqueeze(0), w_emb_attr.unsqueeze(0)).item()
        
        # 还计算范数比
        norm_lm = w_lm_attr.norm().item()
        norm_emb = w_emb_attr.norm().item()
        norm_ratio = norm_lm / norm_emb if norm_emb > 0 else 0
        
        attr_alignment[attr] = {
            "cos_wlm_wembed": round(cos_attr, 6),
            "norm_lm": round(norm_lm, 4),
            "norm_emb": round(norm_emb, 4),
            "norm_ratio": round(norm_ratio, 4),
        }
    
    # 按类别汇总
    for cat_name, cat_attrs in STIMULI.items():
        cos_vals = [attr_alignment[a]["cos_wlm_wembed"] for a in cat_attrs if a in attr_alignment]
        if cos_vals:
            L.log(f"  {cat_name}: avg_cos={np.mean(cos_vals):.4f}, min={min(cos_vals):.4f}, max={max(cos_vals):.4f}")
    
    # 特殊属性详细
    for attr in ["red","green","blue","sweet","sour","hot","cold","big","small"]:
        if attr in attr_alignment:
            a = attr_alignment[attr]
            L.log(f"  {attr}: cos={a['cos_wlm_wembed']}, ||W_lm||={a['norm_lm']}, ||W_emb||={a['norm_emb']}, ratio={a['norm_ratio']}")
    
    return {
        "is_tied": is_tied,
        "global_avg_cos": round(avg_cos, 6),
        "global_median_cos": round(median_cos, 6),
        "attr_alignment": attr_alignment,
    }

# ===================== P360: 逐层信号保真度 =====================
def run_p360(model, tokenizer, device, model_name):
    """
    逐层计算: 方向保持率 = cos(Δh_L, W_lm[attr]) / cos(Δh_{L-1}, W_lm[attr])
    
    这告诉我们每层是"保持方向"还是"扭曲方向"
    """
    L.log("=== P360: 逐层信号保真度 ===")
    
    method = "lm_head"
    beta = 8.0
    n_layers = len(model.model.layers) if hasattr(model, 'model') and hasattr(model.model, 'layers') else 0
    D = model.config.hidden_size
    
    test_attrs = ["red","green","blue","sweet","sour","hot","cold","big","small","long","wide","soft"]
    test_nouns = NOUNS[:10]
    
    results = {}
    
    for ai, attr in enumerate(test_attrs):
        direction, attr_tok_id = get_attr_direction(model, tokenizer, attr, method)
        if direction is None:
            continue
        
        L.log(f"  [{ai+1}/12] {attr}")
        
        lm_head = model.lm_head if hasattr(model, 'lm_head') else model.get_output_embeddings()
        w_lm_attr = lm_head.weight[attr_tok_id].detach().cpu().float()
        w_lm_attr_norm = w_lm_attr / w_lm_attr.norm()
        w_lm_attr_norm_np = w_lm_attr_norm.numpy()
        
        # 收集每个noun的逐层cos
        all_layer_cos = {li: [] for li in range(n_layers)}
        
        for noun in test_nouns:
            prompt = f"The {noun} is"
            toks = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = toks.input_ids
            seq_len = input_ids.shape[1]
            
            embed_layer = model.get_input_embeddings()
            inputs_embeds_base = embed_layer(input_ids).clone()
            
            direction_tensor = torch.tensor(direction, dtype=inputs_embeds_base.dtype, device=device)
            inputs_embeds_intervened = inputs_embeds_base.clone()
            inputs_embeds_intervened[0, -1, :] = inputs_embeds_intervened[0, -1, :] + beta * direction_tensor
            
            # Hook收集各层hidden state
            base_hidden = {}
            interv_hidden = {}
            
            def make_hook(storage, key):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        storage[key] = output[0].detach().float().cpu()
                    else:
                        storage[key] = output.detach().float().cpu()
                return hook
            
            # Base forward
            hooks = []
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                for li in range(n_layers):
                    layer = model.model.layers[li]
                    hooks.append(layer.register_forward_hook(make_hook(base_hidden, f"L{li}")))
            
            with torch.no_grad():
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                try:
                    _ = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids)
                except:
                    for h in hooks: h.remove()
                    continue
            for h in hooks: h.remove()
            
            # Intervened forward
            hooks2 = []
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                for li in range(n_layers):
                    layer = model.model.layers[li]
                    hooks2.append(layer.register_forward_hook(make_hook(interv_hidden, f"L{li}")))
            
            with torch.no_grad():
                try:
                    _ = model(inputs_embeds=inputs_embeds_intervened, position_ids=position_ids)
                except:
                    for h in hooks2: h.remove()
                    continue
            for h in hooks2: h.remove()
            
            # 计算逐层cos
            for li in range(n_layers):
                key = f"L{li}"
                if key not in base_hidden or key not in interv_hidden:
                    continue
                h_base = base_hidden[key][0, -1, :].numpy()
                h_interv = interv_hidden[key][0, -1, :].numpy()
                delta_h = h_interv - h_base
                delta_norm = np.linalg.norm(delta_h)
                if delta_norm > 1e-8:
                    cos_val = float(np.dot(delta_h, w_lm_attr_norm_np) / delta_norm)
                else:
                    cos_val = 0.0
                all_layer_cos[li].append(cos_val)
        
        # 汇总逐层保真度
        avg_cos_per_layer = {}
        for li in range(n_layers):
            if all_layer_cos[li]:
                avg_cos_per_layer[li] = round(np.mean(all_layer_cos[li]), 4)
            else:
                avg_cos_per_layer[li] = 0.0
        
        # 计算逐层保真度（当前层/上一层）
        fidelity_per_layer = {}
        for li in range(1, n_layers):
            prev = avg_cos_per_layer.get(li-1, 0)
            curr = avg_cos_per_layer.get(li, 0)
            if abs(prev) > 1e-6:
                fidelity_per_layer[li] = round(curr / prev, 4)
            else:
                fidelity_per_layer[li] = 0.0
        
        # 找关键层
        # 保真度最低的层 = 扭曲最严重的层
        if fidelity_per_layer:
            min_fid_layer = min(fidelity_per_layer, key=fidelity_per_layer.get)
            min_fid_val = fidelity_per_layer[min_fid_layer]
        else:
            min_fid_layer, min_fid_val = -1, 0
        
        results[attr] = {
            "avg_cos_per_layer": avg_cos_per_layer,
            "fidelity_per_layer": fidelity_per_layer,
            "min_fidelity_layer": min_fid_layer,
            "min_fidelity_value": min_fid_val,
        }
        
        L.log(f"    L0 cos={avg_cos_per_layer.get(0,'N/A')}, "
              f"min_fid@L{min_fid_layer}={min_fid_val}, "
              f"final cos={avg_cos_per_layer.get(n_layers-1,'N/A')}")
    
    return results

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

# ===================== P361: 多方向正交性与组合干预 =====================
def run_p361(model, tokenizer, device, model_name):
    """
    1. 计算W_lm[attr1]和W_lm[attr2]之间的cos
    2. 寻找正交属性对(cos≈0)
    3. 测试正交属性对的同时干预效果
    """
    L.log("=== P361: 多方向正交性与组合干预 ===")
    
    lm_head = model.lm_head if hasattr(model, 'lm_head') else model.get_output_embeddings()
    W_lm = lm_head.weight.detach().float()  # [vocab, D]
    D = W_lm.shape[1]
    
    # 获取所有属性词的W_lm方向
    attr_dirs = {}
    for attr in ALL_ATTRS:
        attr_tok_ids = tokenizer.encode(attr, add_special_tokens=False)
        if len(attr_tok_ids) == 0:
            continue
        tok_id = attr_tok_ids[0]
        if tok_id >= W_lm.shape[0]:
            continue
        w = W_lm[tok_id]
        w_norm = F.normalize(w.unsqueeze(0), dim=1).squeeze(0)
        attr_dirs[attr] = {"dir": w_norm, "tok_id": tok_id}
    
    # 计算属性对之间的cos
    L.log(f"  计算属性对之间的cos... ({len(attr_dirs)}个属性)")
    attr_list = list(attr_dirs.keys())
    cos_matrix = {}
    
    for i, a1 in enumerate(attr_list):
        for j, a2 in enumerate(attr_list):
            if i >= j:
                continue
            cos_val = F.cosine_similarity(
                attr_dirs[a1]["dir"].unsqueeze(0),
                attr_dirs[a2]["dir"].unsqueeze(0)
            ).item()
            cos_matrix[(a1, a2)] = round(cos_val, 4)
    
    # 找最正交的属性对(同一类别内)
    same_cat_pairs = {}
    cross_cat_pairs = {}
    
    for (a1, a2), cos_val in cos_matrix.items():
        cat1 = None
        cat2 = None
        for cat, attrs in STIMULI.items():
            if a1 in attrs: cat1 = cat
            if a2 in attrs: cat2 = cat
        
        if cat1 == cat2:
            same_cat_pairs[(a1, a2)] = cos_val
        else:
            cross_cat_pairs[(a1, a2)] = cos_val
    
    # 找跨类别最正交的top-5对
    sorted_cross = sorted(cross_cat_pairs.items(), key=lambda x: abs(x[1]))
    L.log(f"  跨类别最正交top-5:")
    for (a1, a2), cos_val in sorted_cross[:5]:
        L.log(f"    {a1}-{a2}: cos={cos_val}")
    
    # 找跨类别最平行的top-5对
    sorted_parallel = sorted(cross_cat_pairs.items(), key=lambda x: -abs(x[1]))
    L.log(f"  跨类别最平行top-5:")
    for (a1, a2), cos_val in sorted_parallel[:5]:
        L.log(f"    {a1}-{a2}: cos={cos_val}")
    
    # 测试正交属性对的同时干预
    beta = 8.0
    n_layers = len(model.model.layers) if hasattr(model, 'model') and hasattr(model.model, 'layers') else 0
    
    # 选择3个正交对和3个平行对进行测试
    test_pairs = []
    for (a1, a2), cos_val in sorted_cross[:3]:
        test_pairs.append((a1, a2, cos_val, "orthogonal"))
    for (a1, a2), cos_val in sorted_parallel[:3]:
        test_pairs.append((a1, a2, cos_val, "parallel"))
    
    L.log(f"  测试组合干预: {len(test_pairs)}对")
    
    combo_results = {}
    
    for a1, a2, cos_12, pair_type in test_pairs:
        L.log(f"    {a1}+{a2} (cos={cos_12}, {pair_type})")
        
        dir1, tok1 = get_attr_direction(model, tokenizer, a1, "lm_head")
        dir2, tok2 = get_attr_direction(model, tokenizer, a2, "lm_head")
        if dir1 is None or dir2 is None:
            continue
        
        success_1 = 0
        success_2 = 0
        success_both = 0
        total = 0
        
        for noun in NOUNS[:10]:
            prompt = f"The {noun} is"
            toks = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = toks.input_ids
            seq_len = input_ids.shape[1]
            
            embed_layer = model.get_input_embeddings()
            inputs_embeds = embed_layer(input_ids).clone()
            
            dir1_t = torch.tensor(dir1, dtype=inputs_embeds.dtype, device=device)
            dir2_t = torch.tensor(dir2, dtype=inputs_embeds.dtype, device=device)
            
            # 同时注入两个方向
            inputs_embeds_intervened = inputs_embeds.clone()
            inputs_embeds_intervened[0, -1, :] += beta * dir1_t + beta * dir2_t
            
            with torch.no_grad():
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                outputs = model(inputs_embeds=inputs_embeds_intervened, position_ids=position_ids)
                logits = outputs.logits[0, -1, :]
            
            # 检查排名
            top20 = logits.topk(20).indices.tolist()
            top1 = logits.topk(1).indices.item()
            
            a1_in_top1 = (tok1 == top1)
            a1_in_top20 = (tok1 in top20)
            a2_in_top20 = (tok2 in top20)
            a2_in_top1 = (tok2 == top1)
            
            if a1_in_top20: success_1 += 1
            if a2_in_top20: success_2 += 1
            if a1_in_top20 and a2_in_top20: success_both += 1
            total += 1
        
        combo_key = f"{a1}+{a2}"
        combo_results[combo_key] = {
            "cos_between": cos_12,
            "pair_type": pair_type,
            "a1_top20_rate": round(success_1/total*100, 1) if total > 0 else 0,
            "a2_top20_rate": round(success_2/total*100, 1) if total > 0 else 0,
            "both_top20_rate": round(success_both/total*100, 1) if total > 0 else 0,
        }
        
        L.log(f"      {a1} top20={success_1}/{total}, {a2} top20={success_2}/{total}, both={success_both}/{total}")
    
    return {
        "cos_matrix_sample": {f"{a1}+{a2}": v for (a1, a2), v in list(cos_matrix.items())[:30]},
        "combo_results": combo_results,
    }

# ===================== 主函数 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    model_name = args.model
    L.log(f"===== Phase LXVII: W_lm/W_embed对齐度验证 =====")
    L.log(f"模型: {model_name}")
    
    L.log("加载模型...")
    model, tokenizer, device = load_model(model_name)
    n_layers = len(model.model.layers) if hasattr(model, 'model') and hasattr(model.model, 'layers') else 0
    D = model.config.hidden_size
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    
    L.log(f"模型: {model_name}, 层数: {n_layers}, 维度: {D}, 参数量: {n_params:.1f}B")
    
    # P359: W_lm与W_embed对齐度 ★★★
    L.log("P359: W_lm与W_embed对齐度验证...")
    p359_results = run_p359(model, tokenizer, device, model_name)
    
    # P360: 逐层信号保真度
    L.log("P360: 逐层信号保真度...")
    p360_results = run_p360(model, tokenizer, device, model_name)
    
    # P361: 多方向正交性
    L.log("P361: 多方向正交性与组合干预...")
    p361_results = run_p361(model, tokenizer, device, model_name)
    
    # 保存结果
    all_results = {
        "model": model_name,
        "n_layers": n_layers,
        "D": D,
        "n_params_B": round(n_params, 2),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "p359_wlm_wembed_alignment": p359_results,
        "p360_layer_fidelity": p360_results,
        "p361_orthogonality": p361_results,
    }
    
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = OUT_DIR / f"phase_lxvii_p359_361_{model_name}_{ts}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    L.log(f"结果已保存: {out_path}")
    
    # 汇总
    L.log("\n===== 汇总 =====")
    
    L.log("\nP359: W_lm与W_embed对齐度")
    L.log(f"  tied={p359_results['is_tied']}, global_avg_cos={p359_results['global_avg_cos']}")
    for attr in ["red","green","blue","sweet","sour","hot","cold","big","small"]:
        if attr in p359_results['attr_alignment']:
            a = p359_results['attr_alignment'][attr]
            L.log(f"  {attr}: cos(W_lm, W_embed)={a['cos_wlm_wembed']}")
    
    L.log("\nP360: 逐层信号保真度")
    for attr in ["red","sweet","hot","big"]:
        if attr in p360_results:
            r = p360_results[attr]
            cos0 = r['avg_cos_per_layer'].get(0, 'N/A')
            min_l = r['min_fidelity_layer']
            min_v = r['min_fidelity_value']
            L.log(f"  {attr}: L0 cos={cos0}, min_fid@L{min_l}={min_v}")
    
    L.log("\nP361: 组合干预")
    for key, val in p361_results.get('combo_results', {}).items():
        L.log(f"  {key} ({val['pair_type']}): {val['a1_top20_rate']}%/{val['a2_top20_rate']}% both={val['both_top20_rate']}%")
    
    # 释放GPU
    del model
    gc.collect()
    torch.cuda.empty_cache()
    L.log("GPU已释放")
    L.close()

if __name__ == "__main__":
    main()
