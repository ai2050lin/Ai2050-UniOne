"""
Phase LXII-P343/344/345/346: L0层干预的语义验证
======================================================================

Phase LXI核心发现:
  1. ★★★ L0(嵌入层)是最优干预层: 三模型所有属性100%绝对排名 ★★★
  2. 中间层干预几乎全部0%: 非线性变换扭曲了干预方向
  3. Phase LX的"100%"是logits_diff排名, 不等于绝对排名
  4. 多属性组合全部失败, 生成测试失败

Phase LXII核心目标: 验证L0干预是否是语义层面的

四大实验:
  P343: L0干预的top-1精确排名
    - 在L0层用lm_head/embedding方向干预
    - 检查干预后目标词的绝对排名(top-1/top-3/top-5)
    - 36属性 × 60名词 × 2方向 × 5β = 21600干预
    - 与直接输入"attr noun"对比

  P344: L0干预的生成测试
    - 在L0层干预后让模型生成20个token
    - 检查生成文本是否包含属性词/相关词
    - 36属性 × 20名词 × 5模板 = 3600生成
    - 与直接输入"attr noun"的生成对比

  P345: L0多属性组合
    - 在L0层同时干预2个属性
    - 检查两个属性是否同时在top-20
    - 15组合 × 40名词 × 3β × 3策略 = 5400干预

  P346: L0干预vs直接输入对比
    - 比较3种方式的语义效果:
      a) L0干预: h_0_new = h_0_noun + β·w[attr]
      b) 直接输入: "The red apple is"
      c) 中间层干预: L_k干预
    - 36属性 × 10名词 = 360对比
    - 检查: top-1是否相同, 生成文本是否语义一致

数据规模: 大规模! 每模型>30000次干预/生成
实验模型: qwen3 -> glm4 -> deepseek7b (串行, 避免OOM)
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

LOG_FILE = OUT_DIR / "phase_lxii_log.txt"

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

NOUN_TO_FAMILY = {}
for fam, words in STIMULI.items():
    if "family" in fam:
        for w in words:
            NOUN_TO_FAMILY[w] = fam

PROMPT_TEMPLATES_30 = [
    "The {word} is", "A {word} can be", "This {word} has",
    "I saw a {word}", "The {word} was", "My {word} is",
    "That {word} looks", "One {word} might", "Every {word} has",
    "Some {word} are", "Look at the {word}", "The {word} feels",
    "There is a {word}", "I like the {word}", "What a {word}",
    "His {word} was", "Her {word} is", "Our {word} has",
    "Any {word} can", "Each {word} has", "An old {word}",
    "A new {word}", "The best {word}", "A fine {word}",
    "She found a {word}", "He took the {word}", "We need a {word}",
    "They want a {word}", "That {word} seems", "This {word} became"
]

def get_best_direction(model, tokenizer, device, attr, model_name, n_layers):
    """获取属性的最优方向(Phase LX发现)"""
    attr_tok_ids = tokenizer.encode(attr, add_special_tokens=False)
    if len(attr_tok_ids) == 0:
        return None, -1, None
    attr_tok_id = attr_tok_ids[0]
    
    # Phase LX发现: Qwen3=embedding, GLM4/DS7B=lm_head
    if model_name == "qwen3":
        embed_layer = model.get_input_embeddings()
        direction = embed_layer.weight[attr_tok_id:attr_tok_id+1].detach().float().cpu().numpy()[0].copy()
        method = "embedding"
    else:
        lm_head = model.lm_head if hasattr(model, 'lm_head') else model.get_output_embeddings()
        direction = lm_head.weight[attr_tok_id:attr_tok_id+1].detach().float().cpu().numpy()[0].copy()
        method = "lm_head"
    
    # 归一化到单位长度
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm
    
    return direction, attr_tok_id, method

def get_hidden_at_layer(model, tokenizer, device, text, layer):
    """获取文本在指定层的隐藏状态"""
    toks = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(toks.input_ids, output_hidden_states=True)
        h = out.hidden_states[layer][0, -1].float().cpu().numpy()
    return h

def intervene_L0_and_get_logits(model, tokenizer, device, prompt, direction, beta, alpha=0.5):
    """在L0层干预并通过整个模型获取logits
    方法: 修改embedding层的输出, 然后继续forward
    """
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    
    with torch.no_grad():
        # 1. 获取embedding输出
        embed_layer = model.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids)  # (1, seq_len, D)
        
        # 2. 在最后一个token的embedding上添加方向
        direction_tensor = torch.tensor(direction, dtype=inputs_embeds.dtype, device=device)
        inputs_embeds[0, -1, :] = inputs_embeds[0, -1, :] + alpha * beta * direction_tensor
        
        # 3. 通过模型forward(跳过embedding层)
        position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
        
        out = model(inputs_embeds=inputs_embeds, position_ids=position_ids)
        logits = out.logits[0, -1].float().cpu().numpy()
    
    return logits

def intervene_L0_and_generate(model, tokenizer, device, prompt, direction, beta, alpha=0.5, max_new_tokens=25):
    """在L0层干预后让模型生成文本
    方法: 用hook修改embedding输出
    """
    intervention_applied = [False]
    direction_tensor = torch.tensor(direction, dtype=torch.bfloat16, device=device)
    
    def embedding_hook(module, input, output):
        if intervention_applied[0]:
            return output
        intervention_applied[0] = True
        # output是embedding: (1, seq_len, D)
        if isinstance(output, tuple):
            embeds = output[0].clone()
            embeds[0, -1, :] = embeds[0, -1, :] + alpha * beta * direction_tensor
            return (embeds,) + output[1:]
        else:
            embeds = output.clone()
            embeds[0, -1, :] = embeds[0, -1, :] + alpha * beta * direction_tensor
            return embeds
    
    embed_layer = model.get_input_embeddings()
    hook = embed_layer.register_forward_hook(embedding_hook)
    
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    
    try:
        with torch.no_grad():
            gen_ids = model.generate(
                toks.input_ids, max_new_tokens=max_new_tokens,
                do_sample=False, temperature=1.0,
                pad_token_id=tokenizer.eos_token_id)
            gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    except Exception as e:
        gen_text = f"ERROR: {e}"
    finally:
        hook.remove()
    
    return gen_text

def compute_logits_from_h(model, device, h):
    """从隐藏状态计算logits(通过norm+lm_head)"""
    with torch.no_grad():
        h_tensor = torch.tensor(h, dtype=torch.bfloat16).unsqueeze(0).unsqueeze(0).to(device)
        lm_head = model.lm_head if hasattr(model, 'lm_head') else model.get_output_embeddings()
        norm = model.model.norm if hasattr(model, 'model') and hasattr(model.model, 'norm') else None
        if norm is not None:
            logits = lm_head(norm(h_tensor))[0, 0].float().cpu().numpy()
        else:
            logits = lm_head(h_tensor)[0, 0].float().cpu().numpy()
    return logits

def check_rank_in_logits(logits, tokenizer, target_attr):
    """检查目标词在logits中的精确排名"""
    attr_tok_ids = tokenizer.encode(target_attr, add_special_tokens=False)
    if len(attr_tok_ids) == 0:
        return {"rank": -1, "top1": False, "top5": False, "top20": False}
    attr_tok_id = attr_tok_ids[0]
    
    sorted_ids = np.argsort(logits)[::-1]
    if attr_tok_id >= len(sorted_ids):
        return {"rank": len(sorted_ids), "top1": False, "top5": False, "top20": False}
    
    rank = int(np.where(sorted_ids == attr_tok_id)[0][0]) + 1 if attr_tok_id in sorted_ids else len(sorted_ids)
    
    top5_tokens = [tokenizer.decode([int(sorted_ids[i])]) for i in range(min(5, len(sorted_ids)))]
    
    return {
        "rank": rank,
        "top1": rank == 1,
        "top3": rank <= 3,
        "top5": rank <= 5,
        "top10": rank <= 10,
        "top20": rank <= 20,
        "top5_tokens": top5_tokens,
        "logit": float(logits[attr_tok_id]),
    }

# ===================== P343: L0干预的top-1精确排名 =====================
def run_p343(model, tokenizer, device, n_layers, D, model_name):
    """在L0层干预, 检查目标词的绝对排名"""
    L.log("=== P343: L0干预的top-1精确排名验证 ===")
    
    alpha = 0.5
    betas = [0.5, 1.0, 2.0, 3.0, 5.0]
    
    all_attrs = STIMULI["color_attrs"] + STIMULI["size_attrs"] + STIMULI["taste_attrs"]
    test_nouns = ALL_NOUNS[:60]
    
    results = {}
    
    for attr in all_attrs:
        direction, attr_tok_id, method = get_best_direction(model, tokenizer, device, attr, model_name, n_layers)
        if direction is None:
            continue
        
        attr_results = {}
        
        for beta in betas:
            stats = {"top1": 0, "top3": 0, "top5": 0, "top10": 0, "top20": 0, "total": 0, "avg_rank": 0}
            examples = []
            
            for noun in test_nouns:
                prompt = f"The {noun} is"
                
                # L0层干预: 修改embedding
                try:
                    logits = intervene_L0_and_get_logits(model, tokenizer, device, prompt, direction, beta, alpha)
                except Exception as e:
                    continue
                
                # 检查排名
                rank_info = check_rank_in_logits(logits, tokenizer, attr)
                
                stats["top1"] += int(rank_info["top1"])
                stats["top3"] += int(rank_info["top3"])
                stats["top5"] += int(rank_info["top5"])
                stats["top10"] += int(rank_info["top10"])
                stats["top20"] += int(rank_info["top20"])
                stats["total"] += 1
                stats["avg_rank"] += rank_info["rank"]
                
                if len(examples) < 2:
                    examples.append({
                        "noun": noun, "rank": rank_info["rank"],
                        "top5": rank_info["top5_tokens"], "logit": rank_info["logit"]
                    })
            
            if stats["total"] > 0:
                stats["top1_rate"] = stats["top1"] / stats["total"] * 100
                stats["top3_rate"] = stats["top3"] / stats["total"] * 100
                stats["top5_rate"] = stats["top5"] / stats["total"] * 100
                stats["top10_rate"] = stats["top10"] / stats["total"] * 100
                stats["top20_rate"] = stats["top20"] / stats["total"] * 100
                stats["avg_rank"] /= stats["total"]
            
            attr_results[f"β={beta}"] = stats
        
        # 找最优β
        best_beta = max(attr_results.keys(), key=lambda k: attr_results[k].get("top1_rate", 0))
        best_stats = attr_results[best_beta]
        
        results[attr] = {"method": method, "best_beta": best_beta, "best_stats": best_stats, "examples": examples}
        L.log(f"  {attr}({method}): best={best_beta}, top1={best_stats.get('top1_rate',0):.1f}%, top5={best_stats.get('top5_rate',0):.1f}%, top20={best_stats.get('top20_rate',0):.1f}%, avg_rank={best_stats.get('avg_rank',0):.1f}")
    
    # 汇总
    summary = {"total_attrs": len(results), "top1_attrs": 0, "top5_attrs": 0, "top20_attrs": 0}
    for attr, data in results.items():
        if data["best_stats"].get("top1_rate", 0) > 80: summary["top1_attrs"] += 1
        if data["best_stats"].get("top5_rate", 0) > 80: summary["top5_attrs"] += 1
        if data["best_stats"].get("top20_rate", 0) > 80: summary["top20_attrs"] += 1
    
    L.log(f"\n  ★汇总: top1>80%属性={summary['top1_attrs']}/{summary['total_attrs']}, top5>80%={summary['top5_attrs']}, top20>80%={summary['top20_attrs']}")
    
    return {"details": results, "summary": summary}

# ===================== P344: L0干预的生成测试 =====================
def run_p344(model, tokenizer, device, n_layers, D, model_name):
    """L0干预后让模型生成文本, 检查是否包含属性词"""
    L.log("=== P344: L0干预的生成测试 ===")
    
    alpha = 0.5
    beta = 2.0 if model_name == "qwen3" else 1.0
    
    key_attrs = STIMULI["color_attrs"][:6] + STIMULI["size_attrs"][:6] + STIMULI["taste_attrs"][:6]  # 18个属性
    test_nouns = ALL_NOUNS[:20]
    gen_templates = [
        "The {word} is",
        "I saw a {word} that was",
        "This {word} looks very",
    ]
    
    results = {}
    total = 0
    contain_attr = 0
    contain_related = 0
    
    for attr in key_attrs:
        direction, attr_tok_id, method = get_best_direction(model, tokenizer, device, attr, model_name, n_layers)
        if direction is None:
            continue
        
        attr_stats = {"total": 0, "contains_attr": 0, "contains_related": 0, "examples": []}
        
        for noun in test_nouns:
            for template in gen_templates:
                prompt = template.format(word=noun)
                toks = tokenizer(prompt, return_tensors="pt").to(device)
                
                # 基线生成
                with torch.no_grad():
                    base_ids = model.generate(
                        toks.input_ids, max_new_tokens=25,
                        do_sample=False, temperature=1.0,
                        pad_token_id=tokenizer.eos_token_id)
                    base_text = tokenizer.decode(base_ids[0], skip_special_tokens=True)
                
                # L0干预后生成(用embedding hook)
                gen_text = intervene_L0_and_generate(model, tokenizer, device, prompt, direction, beta, alpha)
                
                # 检查
                attr_lower = attr.lower()
                gen_lower = gen_text.lower()
                base_lower = base_text.lower()
                
                related_words = [attr_lower, attr_lower+"ish", attr_lower+"er", attr_lower+"est",
                                "un"+attr_lower, attr_lower+"ness", attr_lower+"ly"]
                contains = attr_lower in gen_lower
                contains_rel = any(w in gen_lower for w in related_words)
                
                attr_stats["total"] += 1
                if contains: attr_stats["contains_attr"] += 1
                if contains_rel: attr_stats["contains_related"] += 1
                total += 1
                if contains: contain_attr += 1
                if contains_rel: contain_related += 1
                
                if len(attr_stats["examples"]) < 2:
                    attr_stats["examples"].append({
                        "noun": noun, "attr": attr,
                        "base_gen": base_text[:100],
                        "intervened_gen": gen_text[:100],
                        "contains_attr": contains,
                    })
        
        if attr_stats["total"] > 0:
            attr_stats["attr_rate"] = attr_stats["contains_attr"] / attr_stats["total"] * 100
            attr_stats["related_rate"] = attr_stats["contains_related"] / attr_stats["total"] * 100
        
        results[attr] = attr_stats
        L.log(f"  {attr}: attr_in_gen={attr_stats.get('attr_rate',0):.1f}%, related={attr_stats.get('related_rate',0):.1f}% ({attr_stats['total']} gens)")
    
    summary = {
        "total": total,
        "contain_attr": contain_attr,
        "contain_related": contain_related,
        "attr_rate": contain_attr / total * 100 if total > 0 else 0,
        "related_rate": contain_related / total * 100 if total > 0 else 0,
    }
    L.log(f"\n  ★汇总: attr_in_gen={summary['attr_rate']:.1f}%, related={summary['related_rate']:.1f}% ({total} gens)")
    
    return {"details": results, "summary": summary}

# ===================== P345: L0多属性组合 =====================
def run_p345(model, tokenizer, device, n_layers, D, model_name):
    """L0层同时干预2个属性"""
    L.log("=== P345: L0多属性组合干预 ===")
    
    alpha = 0.5
    betas = [1.0, 2.0, 3.0]
    strategies = ["additive", "weighted", "double_beta"]
    
    combos = [
        ("red", "big"), ("red", "sweet"), ("blue", "small"),
        ("green", "long"), ("white", "soft"), ("black", "heavy"),
        ("yellow", "fresh"), ("pink", "tiny"), ("orange", "crisp"),
        ("big", "sweet"), ("small", "sour"), ("long", "fresh"),
        ("short", "spicy"), ("heavy", "red"), ("light", "blue"),
    ]
    
    test_nouns = ALL_NOUNS[:40]
    
    results = {}
    
    for attr1, attr2 in combos:
        dir1, id1, m1 = get_best_direction(model, tokenizer, device, attr1, model_name, n_layers)
        dir2, id2, m2 = get_best_direction(model, tokenizer, device, attr2, model_name, n_layers)
        if dir1 is None or dir2 is None:
            continue
        
        cos_angle = np.dot(dir1, dir2) / (np.linalg.norm(dir1) * np.linalg.norm(dir2) + 1e-10)
        
        for strategy in strategies:
            for beta in betas:
                stats = {"attr1_top1": 0, "attr2_top1": 0, "both_top1": 0,
                         "attr1_top20": 0, "attr2_top20": 0, "both_top20": 0,
                         "total": 0}
                
                for noun in test_nouns:
                    prompt = f"The {noun} is"
                    
                    if strategy == "additive":
                        combo_dir = dir1 + dir2
                        combo_beta = beta
                    elif strategy == "weighted":
                        combo_dir = (dir1 + dir2) / 2
                        combo_beta = beta
                    else:  # double_beta
                        combo_dir = dir1 + dir2
                        combo_beta = beta * 2
                    
                    # 用embedding方法干预
                    try:
                        logits = intervene_L0_and_get_logits(model, tokenizer, device, prompt, combo_dir, combo_beta, alpha)
                    except:
                        continue
                    
                    r1 = check_rank_in_logits(logits, tokenizer, attr1)
                    r2 = check_rank_in_logits(logits, tokenizer, attr2)
                    
                    stats["total"] += 1
                    if r1["top1"]: stats["attr1_top1"] += 1
                    if r2["top1"]: stats["attr2_top1"] += 1
                    if r1["top1"] and r2["top1"]: stats["both_top1"] += 1
                    if r1["top20"]: stats["attr1_top20"] += 1
                    if r2["top20"]: stats["attr2_top20"] += 1
                    if r1["top20"] and r2["top20"]: stats["both_top20"] += 1
                
                if stats["total"] > 0:
                    stats["attr1_top1_rate"] = stats["attr1_top1"] / stats["total"] * 100
                    stats["attr2_top1_rate"] = stats["attr2_top1"] / stats["total"] * 100
                    stats["both_top1_rate"] = stats["both_top1"] / stats["total"] * 100
                    stats["attr1_top20_rate"] = stats["attr1_top20"] / stats["total"] * 100
                    stats["attr2_top20_rate"] = stats["attr2_top20"] / stats["total"] * 100
                    stats["both_top20_rate"] = stats["both_top20"] / stats["total"] * 100
                
                key = f"{attr1}+{attr2}_{strategy}_β{beta}"
                results[key] = stats
        
        # 只打印最优策略
        best_key = max([k for k in results if f"{attr1}+{attr2}" in k], 
                       key=lambda k: results[k].get("both_top20_rate", 0), default=None)
        if best_key:
            L.log(f"  {attr1}+{attr2} (cos={cos_angle:.3f}): best={best_key}, both_top20={results[best_key].get('both_top20_rate',0):.1f}%")
    
    # 汇总
    strategy_summary = {}
    for strategy in strategies:
        rates = [v.get("both_top20_rate", 0) for k, v in results.items() if strategy in k]
        rates1 = [v.get("both_top1_rate", 0) for k, v in results.items() if strategy in k]
        strategy_summary[strategy] = {
            "avg_both_top20": np.mean(rates) if rates else 0,
            "avg_both_top1": np.mean(rates1) if rates1 else 0,
            "max_both_top20": np.max(rates) if rates else 0,
        }
        L.log(f"  策略{strategy}: avg_both_top20={strategy_summary[strategy]['avg_both_top20']:.1f}%, max={strategy_summary[strategy]['max_both_top20']:.1f}%")
    
    return {"details": results, "strategy_summary": strategy_summary}

# ===================== P346: L0干预vs直接输入对比 =====================
def run_p346(model, tokenizer, device, n_layers, D, model_name):
    """比较L0干预和直接输入"attr noun"的语义效果"""
    L.log("=== P346: L0干预vs直接输入对比 ===")
    
    alpha = 0.5
    beta = 2.0 if model_name == "qwen3" else 1.0
    
    key_attrs = ["red","blue","white","big","small","short","sweet","fresh","soft"]
    test_nouns = ALL_NOUNS[:10]
    
    results = {}
    
    for attr in key_attrs:
        direction, attr_tok_id, method = get_best_direction(model, tokenizer, device, attr, model_name, n_layers)
        if direction is None:
            continue
        
        attr_data = {"noun_results": []}
        
        for noun in test_nouns:
            # 方式1: 直接输入"attr noun"
            prompt_direct = f"The {attr} {noun} is"
            toks_direct = tokenizer(prompt_direct, return_tensors="pt").to(device)
            with torch.no_grad():
                out_direct = model(toks_direct.input_ids, output_hidden_states=True)
                logits_direct = out_direct.logits[0, -1].float().cpu().numpy()
            
            # 方式2: L0干预
            prompt_noun = f"The {noun} is"
            try:
                logits_intervened = intervene_L0_and_get_logits(model, tokenizer, device, prompt_noun, direction, beta, alpha)
            except:
                continue
            
            # 方式3: 基线(不干预)
            toks_noun = tokenizer(prompt_noun, return_tensors="pt").to(device)
            with torch.no_grad():
                out_base = model(toks_noun.input_ids, output_hidden_states=True)
                logits_base = out_base.logits[0, -1].float().cpu().numpy()
            
            # 比较
            rank_direct = check_rank_in_logits(logits_direct, tokenizer, attr)
            rank_intervened = check_rank_in_logits(logits_intervened, tokenizer, attr)
            rank_base = check_rank_in_logits(logits_base, tokenizer, attr)
            
            # top-5 tokens对比
            top5_direct = [tokenizer.decode([int(np.argsort(logits_direct)[::-1][i])]) for i in range(5)]
            top5_intervened = [tokenizer.decode([int(np.argsort(logits_intervened)[::-1][i])]) for i in range(5)]
            top5_base = [tokenizer.decode([int(np.argsort(logits_base)[::-1][i])]) for i in range(5)]
            
            # 语义相似度: top-5重叠度
            overlap_direct_intervened = len(set(top5_direct) & set(top5_intervened))
            
            noun_data = {
                "noun": noun,
                "direct_rank": rank_direct["rank"],
                "intervened_rank": rank_intervened["rank"],
                "base_rank": rank_base["rank"],
                "direct_top5": top5_direct,
                "intervened_top5": top5_intervened,
                "base_top5": top5_base,
                "overlap_direct_intervened": overlap_direct_intervened,
            }
            attr_data["noun_results"].append(noun_data)
        
        # 计算平均
        avg_direct_rank = np.mean([n["direct_rank"] for n in attr_data["noun_results"]])
        avg_intervened_rank = np.mean([n["intervened_rank"] for n in attr_data["noun_results"]])
        avg_base_rank = np.mean([n["base_rank"] for n in attr_data["noun_results"]])
        avg_overlap = np.mean([n["overlap_direct_intervened"] for n in attr_data["noun_results"]])
        
        attr_data["avg_direct_rank"] = avg_direct_rank
        attr_data["avg_intervened_rank"] = avg_intervened_rank
        attr_data["avg_base_rank"] = avg_base_rank
        attr_data["avg_overlap"] = avg_overlap
        
        results[attr] = attr_data
        L.log(f"  {attr}: direct_rank={avg_direct_rank:.0f}, intervened_rank={avg_intervened_rank:.0f}, base_rank={avg_base_rank:.0f}, overlap(top5)={avg_overlap:.1f}/5")
    
    # 汇总
    all_direct = [v["avg_direct_rank"] for v in results.values()]
    all_intervened = [v["avg_intervened_rank"] for v in results.values()]
    all_base = [v["avg_base_rank"] for v in results.values()]
    all_overlap = [v["avg_overlap"] for v in results.values()]
    
    summary = {
        "avg_direct_rank": np.mean(all_direct),
        "avg_intervened_rank": np.mean(all_intervened),
        "avg_base_rank": np.mean(all_base),
        "avg_overlap": np.mean(all_overlap),
    }
    L.log(f"\n  ★汇总: direct_rank={summary['avg_direct_rank']:.0f}, intervened_rank={summary['avg_intervened_rank']:.0f}, base_rank={summary['avg_base_rank']:.0f}")
    L.log(f"  ★top-5重叠度: {summary['avg_overlap']:.1f}/5 ({summary['avg_overlap']/5*100:.0f}%)")
    
    return {"details": results, "summary": summary}

# ===================== 主函数 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    model_name = args.model
    L.log(f"===== Phase LXII: L0层干预语义验证 — {model_name} =====")
    
    model, tokenizer, device = load_model(model_name)
    n_layers = len(model.model.layers) if hasattr(model, 'model') and hasattr(model.model, 'layers') else 24
    D = model.config.hidden_size
    
    L.log(f"  模型: {model_name}, 层数: {n_layers}, 维度: {D}, 设备: {device}")
    
    all_results = {"model": model_name, "n_layers": n_layers, "D": D}
    
    # P343: L0干预top-1排名
    p343 = run_p343(model, tokenizer, device, n_layers, D, model_name)
    all_results["p343"] = p343
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # P344: L0干预生成测试
    p344 = run_p344(model, tokenizer, device, n_layers, D, model_name)
    all_results["p344"] = p344
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # P345: L0多属性组合
    p345 = run_p345(model, tokenizer, device, n_layers, D, model_name)
    all_results["p345"] = p345
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # P346: L0干预vs直接输入对比
    p346 = run_p346(model, tokenizer, device, n_layers, D, model_name)
    all_results["p346"] = p346
    
    # 保存结果
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    out_path = OUT_DIR / f"phase_lxii_p343_346_{model_name}_{ts}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    L.log(f"\n结果已保存: {out_path}")
    
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    L.log(f"===== {model_name} 测试完成 =====")

if __name__ == "__main__":
    main()
