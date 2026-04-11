"""
Phase LXV-P355/356: DeepSeek模型规模对比 + P352原理详解
======================================================================

核心目标:
1. 对比DeepSeek 1.5B vs 7B的语义干预效果
2. 重点测试P352矛盾信号在不同规模模型中的表现
3. 验证"信息注入战胜常识"是否是规模依赖的

P355: DeepSeek 1.5B vs 7B 全面对比
  - P352矛盾信号测试
  - P351 Top-1验证
  - P354 β精确调控
  - 36属性 × 60名词 + 30矛盾组合

P356: 规模缩放定律
  - 随模型规模增大, 干预效果如何变化?
  - 最小有效β是否随规模减小?
  - 矛盾信号战胜率是否随规模增加?

实验模型: deepseek1.5b → deepseek7b (串行, 避免OOM)
"""

import torch
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

LOG_FILE = OUT_DIR / "phase_lxv_log.txt"

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
    """获取模型路径 - 包含DeepSeek 1.5B和7B"""
    paths = {
        "deepseek1.5b": r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B",
        "deepseek7b": r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60",
        "glm4": r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf",
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

# =========== 刺激材料 ===========
STIMULI = {
    "fruit_family": ["apple","banana","pear","orange","grape","mango","strawberry","watermelon","cherry","peach","lemon","lime"],
    "animal_family": ["cat","dog","rabbit","horse","lion","eagle","elephant","dolphin","tiger","bear","fox","wolf"],
    "vehicle_family": ["car","bus","train","plane","boat","bicycle","truck","helicopter","motorcycle","ship","subway","taxi"],
    "furniture_family": ["chair","table","desk","sofa","bed","cabinet","shelf","bench","stool","dresser","couch","armchair"],
    "tool_family": ["hammer","wrench","screwdriver","saw","drill","pliers","knife","scissors","shovel","rake","axe","chisel"],
    "color_attrs": ["red","green","yellow","orange","brown","white","blue","black","pink","purple","gray","gold"],
    "taste_attrs": ["sweet","sour","bitter","salty","crisp","soft","spicy","fresh","tart","savory","rich","mild"],
    "size_attrs": ["big","small","tall","short","long","wide","thin","thick","heavy","light","huge","tiny"],
}

ALL_NOUNS = []
for fam in ["fruit_family","animal_family","vehicle_family","furniture_family","tool_family"]:
    ALL_NOUNS.extend(STIMULI[fam])

ALL_ATTRS = STIMULI["color_attrs"] + STIMULI["taste_attrs"] + STIMULI["size_attrs"]

# 矛盾信号组合
CONTRADICTION_PAIRS = [
    ("fire", "hot", "cold"),
    ("ice", "cold", "hot"),
    ("snow", "cold", "hot"),
    ("sun", "hot", "cold"),
    ("lava", "hot", "cold"),
    ("furnace", "hot", "cold"),
    ("honey", "sweet", "bitter"),
    ("sugar", "sweet", "sour"),
    ("candy", "sweet", "salty"),
    ("lemon", "sour", "sweet"),
    ("vinegar", "sour", "sweet"),
    ("coffee", "bitter", "sweet"),
    ("elephant", "big", "tiny"),
    ("mountain", "big", "small"),
    ("ocean", "big", "tiny"),
    ("ant", "small", "huge"),
    ("grain", "tiny", "big"),
    ("pebble", "small", "huge"),
    ("grass", "green", "red"),
    ("sky", "blue", "black"),
    ("blood", "red", "white"),
    ("night", "black", "white"),
    ("snow", "white", "black"),
    ("gold", "gold", "blue"),
    ("rock", "hard", "soft"),
    ("steel", "hard", "soft"),
    ("pillow", "soft", "hard"),
    ("sponge", "soft", "hard"),
    ("feather", "soft", "hard"),
    ("diamond", "hard", "soft"),
    ("cheetah", "fast", "slow"),
    ("turtle", "slow", "fast"),
    ("rocket", "fast", "slow"),
    ("snail", "slow", "fast"),
]

# =========== 核心函数 ===========
def get_attr_direction(model, tokenizer, attr, method="lm_head"):
    """
    获取属性词的方向向量
    
    ★★★ P352核心公式: 方向向量 = lm_head.weight[token_id] ★★★
    
    数学原理:
      logits[v] = W_lm[v] · norm(h) + bias[v]   (v是词表中的任意词)
      
      如果在h上加上 W_lm[attr] 方向:
        h_new = h + αβ · W_lm[attr] / ||W_lm[attr]||
        logits_new[attr] = W_lm[attr] · norm(h_new) + bias[attr]
                         ≈ logits_base[attr] + αβ · ||W_lm[attr]|| · cos(W_lm[attr], 归一化方向)
        
      因为W_lm[attr]和自己方向完全对齐(cos=1), 所以:
        logits_new[attr] ≈ logits_base[attr] + αβ · ||W_lm[attr]||
        
      → logits[attr]必然增大! → 属性词排名上升!
    """
    attr_tok_ids = tokenizer.encode(attr, add_special_tokens=False)
    if len(attr_tok_ids) == 0:
        return None
    
    if method == "lm_head":
        # ★★★ 核心: 取lm_head权重矩阵的第attr_token_id行 ★★★
        # W_lm ∈ R^{vocab_size × d_model}
        # direction = W_lm[attr_token_id] ∈ R^{d_model}
        lm_head = model.lm_head if hasattr(model, 'lm_head') else model.get_output_embeddings()
        direction = lm_head.weight[attr_tok_ids[0]].detach().cpu().float().numpy().copy()
    elif method == "embedding":
        embed_layer = model.get_input_embeddings()
        direction = embed_layer.weight[attr_tok_ids[0]].detach().cpu().float().numpy().copy()
    else:
        return None
    
    # 归一化: direction / ||direction||
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm
    
    return direction

def intervene_at_position_and_get_logits(model, tokenizer, device, prompt, direction, beta, 
                                          target_positions=None, alpha=1.0):
    """
    ★★★ P352核心操作: 在L0(嵌入层)的last_token位置注入方向 ★★★
    
    完整计算过程:
    
    1. Tokenize: prompt → input_ids (如 "The fire is" → [1, 450, 2564, 374])
    
    2. Embedding: input_ids → inputs_embeds
       inputs_embeds = W_embed[input_ids]  ∈ R^{seq_len × d_model}
       
    3. 干预: 在last_token位置注入方向
       inputs_embeds[0, -1, :] += αβ · direction
       
       对于fire→cold:
         prompt = "The fire is"
         last_token = "is" (位置3)
         direction = W_lm[cold_token_id] / ||W_lm[cold_token_id]||
         
         inputs_embeds[0, 3, :] = W_embed[is] + αβ · W_lm[cold]/||W_lm[cold]||
         
         = 原始"is"的嵌入 + β倍的"cold"方向
         
         = "is"被赋予了"cold"的语义信号!
    
    4. Forward: 修改后的inputs_embeds通过所有Transformer层
       h_L0 = inputs_embeds (已修改)
       h_L1 = Transformer_1(h_L0)
       ...
       h_LN = Transformer_N(h_{LN-1})
       
    5. 计算logits:
       logits = W_lm · norm(h_LN[-1]) + bias
       
       因为h_L0[-1]包含了W_lm[cold]方向, 经过所有层变换后,
       h_LN[-1]仍保留与W_lm[cold]的对齐分量:
       
       logits[cold] = W_lm[cold] · norm(h_LN[-1]) + bias[cold]
                     ≈ W_lm[cold] · (h_base + β·残差·W_lm[cold]) + bias[cold]
                     = logits_base[cold] + β · ||W_lm[cold]||² · cos项
                     > logits_base[cold]  → "cold"排名上升!
    
    6. 对比: 无干预时 logits[hot] > logits[cold] (常识)
             有干预时 logits[cold] > logits[hot] (注入战胜常识!)
    """
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    seq_len = input_ids.shape[1]
    
    with torch.no_grad():
        # Step 1-2: 获取embedding
        embed_layer = model.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids).clone()
        
        # Step 3: 在指定位置注入方向
        direction_tensor = torch.tensor(direction, dtype=inputs_embeds.dtype, device=device)
        
        if target_positions is None:
            # 默认在last_token位置注入
            inputs_embeds[0, -1, :] = inputs_embeds[0, -1, :] + alpha * beta * direction_tensor
        else:
            for pos in target_positions:
                if 0 <= pos < seq_len:
                    inputs_embeds[0, pos, :] = inputs_embeds[0, pos, :] + alpha * beta * direction_tensor
        
        # Step 4: Forward through all layers
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        out = model(inputs_embeds=inputs_embeds, position_ids=position_ids)
        
        # Step 5-6: 获取logits
        logits = out.logits[0, -1].float().cpu().numpy()
    
    return logits

def get_top_k_words(logits, tokenizer, k=10):
    sorted_ids = np.argsort(logits)[::-1][:k]
    words = []
    for tid in sorted_ids:
        word = tokenizer.decode([tid]).strip()
        words.append({"word": word, "token_id": int(tid), "logit": float(logits[tid])})
    return words

def check_rank_in_logits(logits, tokenizer, target_word, top_k=20):
    attr_tok_ids = tokenizer.encode(target_word, add_special_tokens=False)
    if len(attr_tok_ids) == 0:
        return {"hit": False, "rank": -1, "top1": False, "top3": False, "top5": False, "top10": False, "top20": False}
    
    attr_tok_id = attr_tok_ids[0]
    sorted_ids = np.argsort(logits)[::-1]
    rank = int(np.where(sorted_ids == attr_tok_id)[0][0]) + 1 if attr_tok_id in sorted_ids else len(logits)
    
    return {
        "hit": True,
        "rank": rank,
        "top1": rank <= 1,
        "top3": rank <= 3,
        "top5": rank <= 5,
        "top10": rank <= 10,
        "top20": rank <= 20,
    }

# ===================== P355: DeepSeek 1.5B vs 7B 全面对比 =====================
def run_p355(model, tokenizer, device, model_name):
    """
    P355: DeepSeek 1.5B vs 7B的全面对比
    包含: Top-1验证 + 矛盾信号 + β调控
    """
    L.log(f"=== P355: {model_name} 全面对比 ===")
    
    method = "lm_head"
    
    # ---- Part A: Top-1验证 ----
    L.log(f"  Part A: Top-1验证 (36属性 × 60名词 × 5β)")
    beta_range = [1.0, 3.0, 5.0, 8.0, 12.0]
    top1_results = {}
    
    for ai, attr in enumerate(ALL_ATTRS):
        direction = get_attr_direction(model, tokenizer, attr, method)
        if direction is None:
            continue
        
        attr_res = {}
        for beta in beta_range:
            stats = {"top1": 0, "top5": 0, "top10": 0, "top20": 0, "total": 0, "avg_rank": 0}
            
            for noun in ALL_NOUNS[:60]:
                prompt = f"The {noun} is"
                try:
                    logits = intervene_at_position_and_get_logits(
                        model, tokenizer, device, prompt, direction, beta, None)
                except:
                    continue
                
                rank_info = check_rank_in_logits(logits, tokenizer, attr)
                if rank_info["hit"]:
                    stats["total"] += 1
                    stats["avg_rank"] += rank_info["rank"]
                    if rank_info["top1"]: stats["top1"] += 1
                    if rank_info["top5"]: stats["top5"] += 1
                    if rank_info["top10"]: stats["top10"] += 1
                    if rank_info["top20"]: stats["top20"] += 1
            
            if stats["total"] > 0:
                key = f"b{beta}"
                stats["top1%"] = round(stats["top1"] / stats["total"] * 100, 1)
                stats["top20%"] = round(stats["top20"] / stats["total"] * 100, 1)
                stats["avg_rank"] = round(stats["avg_rank"] / stats["total"], 1)
                attr_res[key] = stats
        
        if attr_res:
            best = max(attr_res.keys(), key=lambda k: attr_res[k].get("top20%", 0))
            r = attr_res[best]
            L.log(f"    [{ai+1}/36] {attr}(β={best}): top1%={r['top1%']}, top20%={r['top20%']}, rank={r['avg_rank']}")
        
        top1_results[attr] = attr_res
    
    # ---- Part B: 矛盾信号测试 ----
    L.log(f"  Part B: 矛盾信号测试 (34组合 × 3模板 × 5β)")
    contra_results = {}
    
    for pi, (noun, common_attr, contra_attr) in enumerate(CONTRADICTION_PAIRS):
        contra_direction = get_attr_direction(model, tokenizer, contra_attr, method)
        if contra_direction is None:
            continue
        
        pair_res = {}
        for beta in beta_range:
            stats = {
                "contra_top1": 0, "contra_top5": 0, "contra_top10": 0, "contra_top20": 0,
                "common_top1": 0, "common_top5": 0, "common_top10": 0, "common_top20": 0,
                "total": 0, "contra_avg_rank": 0, "common_avg_rank": 0,
                "examples": []
            }
            
            for template in ["The {word} is", "A {word} can be", "This {word} has"]:
                prompt = template.format(word=noun)
                
                # 基线
                toks = tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    logits_base = model(toks.input_ids).logits[0, -1].float().cpu().numpy()
                
                # 注入矛盾属性方向
                try:
                    logits_contra = intervene_at_position_and_get_logits(
                        model, tokenizer, device, prompt, contra_direction, beta, None)
                except:
                    continue
                
                rank_contra = check_rank_in_logits(logits_contra, tokenizer, contra_attr)
                rank_common = check_rank_in_logits(logits_contra, tokenizer, common_attr)
                
                stats["total"] += 1
                
                if rank_contra["hit"]:
                    stats["contra_avg_rank"] += rank_contra["rank"]
                    if rank_contra["top1"]: stats["contra_top1"] += 1
                    if rank_contra["top5"]: stats["contra_top5"] += 1
                    if rank_contra["top10"]: stats["contra_top10"] += 1
                    if rank_contra["top20"]: stats["contra_top20"] += 1
                
                if rank_common["hit"]:
                    stats["common_avg_rank"] += rank_common["rank"]
                    if rank_common["top1"]: stats["common_top1"] += 1
                    if rank_common["top5"]: stats["common_top5"] += 1
                    if rank_common["top10"]: stats["common_top10"] += 1
                    if rank_common["top20"]: stats["common_top20"] += 1
                
                if len(stats["examples"]) < 1:
                    base_top5 = get_top_k_words(logits_base, tokenizer, 5)
                    contra_top5 = get_top_k_words(logits_contra, tokenizer, 5)
                    stats["examples"].append({
                        "prompt": prompt, "beta": beta,
                        "base_top5": [w["word"] for w in base_top5],
                        "contra_top5": [w["word"] for w in contra_top5],
                        "contra_rank": rank_contra.get("rank", -1),
                        "common_rank": rank_common.get("rank", -1),
                    })
            
            if stats["total"] > 0:
                key = f"b{beta}"
                stats["contra_top1%"] = round(stats["contra_top1"] / stats["total"] * 100, 1)
                stats["contra_top20%"] = round(stats["contra_top20"] / stats["total"] * 100, 1)
                stats["common_top1%"] = round(stats["common_top1"] / stats["total"] * 100, 1)
                stats["common_top20%"] = round(stats["common_top20"] / stats["total"] * 100, 1)
                if stats["contra_top20"] > 0:
                    stats["contra_avg_rank"] = round(stats["contra_avg_rank"] / stats["contra_top20"], 1)
                else:
                    stats["contra_avg_rank"] = -1
                if stats["common_top20"] > 0:
                    stats["common_avg_rank"] = round(stats["common_avg_rank"] / stats["common_top20"], 1)
                else:
                    stats["common_avg_rank"] = -1
                pair_res[key] = stats
        
        if pair_res:
            best = max(pair_res.keys(), key=lambda k: pair_res[k].get("contra_top20%", 0))
            r = pair_res[best]
            L.log(f"    [{pi+1}/34] {noun}({common_attr}→{contra_attr}): "
                  f"contra_top20%={r['contra_top20%']}, common_top20%={r['common_top20%']}")
        
        contra_results[f"{noun}_{common_attr}_vs_{contra_attr}"] = pair_res
    
    # ---- Part C: β精确调控 ----
    L.log(f"  Part C: β精确调控 (12属性 × 20名词 × 17β)")
    beta_sweep = [0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0]
    beta_results = {}
    test_attrs = ["red", "green", "blue", "sweet", "sour", "hot", "cold", "big", "small", "soft", "hard", "fast"]
    
    for ai, attr in enumerate(test_attrs):
        direction = get_attr_direction(model, tokenizer, attr, method)
        if direction is None:
            continue
        
        attr_res = {}
        min_beta_top1 = None
        min_beta_top20 = None
        
        for beta in beta_sweep:
            stats = {"top1": 0, "top5": 0, "top20": 0, "total": 0, "avg_rank": 0}
            
            for noun in ALL_NOUNS[:20]:
                prompt = f"The {noun} is"
                try:
                    logits = intervene_at_position_and_get_logits(
                        model, tokenizer, device, prompt, direction, beta, None)
                except:
                    continue
                
                rank_info = check_rank_in_logits(logits, tokenizer, attr)
                if rank_info["hit"]:
                    stats["total"] += 1
                    stats["avg_rank"] += rank_info["rank"]
                    if rank_info["top1"]: stats["top1"] += 1
                    if rank_info["top5"]: stats["top5"] += 1
                    if rank_info["top20"]: stats["top20"] += 1
            
            if stats["total"] > 0:
                key = f"b{beta}"
                stats["top1%"] = round(stats["top1"] / stats["total"] * 100, 1)
                stats["top5%"] = round(stats["top5"] / stats["total"] * 100, 1)
                stats["top20%"] = round(stats["top20"] / stats["total"] * 100, 1)
                stats["avg_rank"] = round(stats["avg_rank"] / stats["total"], 1)
                attr_res[key] = stats
                
                if stats["top1"] > 0 and min_beta_top1 is None:
                    min_beta_top1 = beta
                if stats["top20"] > 0 and min_beta_top20 is None:
                    min_beta_top20 = beta
        
        L.log(f"    [{ai+1}/12] {attr}: min_β(top1)={min_beta_top1}, min_β(top20)={min_beta_top20}")
        beta_results[attr] = {
            "beta_sweep": attr_res,
            "min_beta_top1": min_beta_top1,
            "min_beta_top20": min_beta_top20,
        }
    
    return {
        "top1_verification": top1_results,
        "contradiction_test": contra_results,
        "beta_precision": beta_results,
    }

# ===================== 主函数 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, 
                        choices=["deepseek1.5b", "deepseek7b", "glm4"])
    args = parser.parse_args()
    
    model_name = args.model
    L.log(f"===== Phase LXV: DeepSeek规模对比 =====")
    L.log(f"模型: {model_name}")
    
    # 加载模型
    L.log("加载模型...")
    model, tokenizer, device = load_model(model_name)
    n_layers = len(model.model.layers) if hasattr(model, 'model') and hasattr(model.model, 'layers') else 0
    D = model.config.hidden_size
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    
    L.log(f"模型: {model_name}, 层数: {n_layers}, 维度: {D}, 参数量: {n_params:.1f}B, 设备: {device}")
    
    # 运行P355
    p355_results = run_p355(model, tokenizer, device, model_name)
    
    # 保存结果
    all_results = {
        "model": model_name,
        "n_layers": n_layers,
        "D": D,
        "n_params_B": round(n_params, 2),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "p355_scale_comparison": p355_results,
    }
    
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = OUT_DIR / f"phase_lxv_p355_{model_name}_{ts}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    L.log(f"结果已保存: {out_path}")
    
    # 汇总
    L.log("\n===== 汇总 =====")
    
    # Top-1汇总
    L.log("\nTop-1验证:")
    for attr in ALL_ATTRS[:12]:
        if attr in p355_results["top1_verification"]:
            attr_res = p355_results["top1_verification"][attr]
            best = max(attr_res.keys(), key=lambda k: attr_res[k].get("top20%", 0)) if attr_res else None
            if best:
                r = attr_res[best]
                L.log(f"  {attr}(β={best}): top1%={r['top1%']}, top20%={r['top20%']}, rank={r['avg_rank']}")
    
    # 矛盾信号汇总
    L.log("\n矛盾信号测试:")
    for key in list(p355_results["contradiction_test"].keys())[:15]:
        pair_res = p355_results["contradiction_test"][key]
        best = max(pair_res.keys(), key=lambda k: pair_res[k].get("contra_top20%", 0)) if pair_res else None
        if best:
            r = pair_res[best]
            L.log(f"  {key}(β={best}): contra_top20%={r['contra_top20%']}, common_top20%={r['common_top20%']}")
    
    # β调控汇总
    L.log("\nβ精确调控:")
    for attr in ["red", "green", "sweet", "sour", "hot", "cold", "soft", "hard"]:
        if attr in p355_results["beta_precision"]:
            r = p355_results["beta_precision"][attr]
            L.log(f"  {attr}: min_β(top1)={r['min_beta_top1']}, min_β(top20)={r['min_beta_top20']}")
    
    # 释放GPU
    del model
    gc.collect()
    torch.cuda.empty_cache()
    L.log("GPU已释放")
    
    L.close()

if __name__ == "__main__":
    main()
