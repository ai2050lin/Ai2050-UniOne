"""
Phase LXI-P339/340/341/342: 干预质量全面验证
======================================================================

Phase LX核心发现:
  1. ★★★ lm_head权重方向: 3模型36属性全部100%! ★★★
  2. Qwen3最优方向=embedding, GLM4/DS7B最优方向=lm_head
  3. β=0.5-3就足够
  4. 但top-20命中≠top-1, 语义质量未验证

Phase LXI核心目标: 验证干预是否是语义层面的

四大实验:
  P339: top-1/top-3/top-5/top-10验证
    - 精确检查目标词在干预后logits中的排名
    - 不只是top-20命中, 而是top-1是否就是目标词
    - 36属性 × 60名词 × 最优方向 = 2160干预

  P340: 生成测试
    - 干预后让模型生成10-20个token
    - 检查生成文本是否包含目标属性词
    - 36属性 × 20名词 × 3模板 = 2160生成

  P341: 多属性组合干预
    - 同时干预2个属性: red+big, sweet+small等
    - 检查两个属性是否同时生效
    - 15组合 × 20名词 × 3β = 900干预

  P342: 跨层稳定性测试
    - 在每层测试最优方向的效果
    - 检查干预是否只在特定层有效
    - 36属性 × 18层 × 20名词 = 12960干预

数据规模: 36属性 × 60名词 × 全层 × 多β × 4实验 × 3模型
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

LOG_FILE = OUT_DIR / "phase_lxi_log.txt"

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

FAMILY_NAMES = ["fruit_family","animal_family","vehicle_family","furniture_family","tool_family"]

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
        return None, -1
    attr_tok_id = attr_tok_ids[0]
    
    # Phase LX发现: Qwen3=embedding, GLM4/DS7B=lm_head
    if model_name == "qwen3":
        embed_layer = model.get_input_embeddings()
        direction = embed_layer.weight[attr_tok_id:attr_tok_id+1].detach().float().cpu().numpy()[0].copy()
    else:
        lm_head = model.lm_head if hasattr(model, 'lm_head') else model.get_output_embeddings()
        direction = lm_head.weight[attr_tok_id:attr_tok_id+1].detach().float().cpu().numpy()[0].copy()
    
    # 归一化到单位长度
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm
    
    return direction, attr_tok_id

def get_best_layer(model_name, n_layers):
    """获取最优层(Phase LX发现)"""
    # Phase LX发现: 大部分属性在最优层100%
    best_layers_map = {
        "qwen3": n_layers // 2 + 2,  # L20附近
        "glm4": n_layers * 3 // 4,    # L30附近
        "deepseek7b": n_layers // 3,   # L10附近
    }
    return best_layers_map.get(model_name, n_layers // 2)

def get_best_beta(model_name):
    """获取最优β(Phase LX发现)"""
    beta_map = {"qwen3": 2.0, "glm4": 1.0, "deepseek7b": 1.0}
    return beta_map.get(model_name, 1.0)

def intervene_with_direction(model, device, h_noun, direction, layer, f_base=None, alpha=0.5, beta=1.0):
    """用任意方向向量干预(含norm层)"""
    h_new = h_noun.copy()
    if f_base is not None:
        h_new = h_new + alpha * f_base
    h_new = h_new + beta * direction
    
    with torch.no_grad():
        h_tensor = torch.tensor(h_new, dtype=torch.bfloat16).unsqueeze(0).unsqueeze(0).to(device)
        h_base_tensor = torch.tensor(h_noun, dtype=torch.bfloat16).unsqueeze(0).unsqueeze(0).to(device)
        
        lm_head = model.lm_head if hasattr(model, 'lm_head') else model.get_output_embeddings()
        norm = None
        if hasattr(model, 'model') and hasattr(model.model, 'norm'):
            norm = model.model.norm
        
        if norm is not None:
            logits_new = lm_head(norm(h_tensor))[0, 0].float().cpu().numpy()
            logits_base = lm_head(norm(h_base_tensor))[0, 0].float().cpu().numpy()
        else:
            logits_new = lm_head(h_tensor)[0, 0].float().cpu().numpy()
            logits_base = lm_head(h_base_tensor)[0, 0].float().cpu().numpy()
    
    logits_diff = logits_new - logits_base
    return logits_new, logits_diff

def check_intervention_rank(logits_new, logits_diff, tokenizer, target_attr):
    """检查目标词的精确排名"""
    attr_tok_ids = tokenizer.encode(target_attr, add_special_tokens=False)
    if len(attr_tok_ids) == 0:
        return {"top1": False, "top3": False, "top5": False, "top10": False, "top20": False, "rank": -1, "logit_change": 0.0}
    attr_tok_id = attr_tok_ids[0]
    
    sorted_ids = np.argsort(logits_new)[::-1]
    rank = int(np.where(sorted_ids == attr_tok_id)[0][0]) + 1 if attr_tok_id in sorted_ids else len(sorted_ids)
    logit_change = float(logits_diff[attr_tok_id]) if attr_tok_id < len(logits_diff) else 0.0
    
    # top-5解码
    top5_tokens = [tokenizer.decode([int(sorted_ids[i])]) for i in range(5)]
    
    return {
        "top1": rank == 1,
        "top3": rank <= 3,
        "top5": rank <= 5,
        "top10": rank <= 10,
        "top20": rank <= 20,
        "rank": rank,
        "logit_change": logit_change,
        "top5_tokens": top5_tokens,
    }

def collect_noun_hidden_states(model, tokenizer, device, nouns, templates, layer):
    """收集名词在指定层的隐藏状态"""
    noun_h = {}
    for noun in nouns:
        h_list = []
        for t in templates:
            text = t.format(word=noun)
            toks = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(toks.input_ids, output_hidden_states=True)
                h_list.append(out.hidden_states[layer][0, -1].float().cpu().numpy())
        noun_h[noun] = np.mean(h_list, axis=0)
    return noun_h

# ===================== P339: top-1/top-3/top-5/top-10精确排名 =====================
def run_p339(model, tokenizer, device, n_layers, D, model_name):
    """验证干预后目标词的精确排名"""
    L.log("=== P339: top-1/top-3/top-5/top-10精确排名验证 ===")
    
    alpha_fixed = 0.5
    beta = get_best_beta(model_name)
    layer = get_best_layer(model_name, n_layers)
    
    L.log(f"  模型: {model_name}, 层: L{layer}, β: {beta}")
    
    # 收集60个名词的隐藏状态
    noun_h = collect_noun_hidden_states(model, tokenizer, device, ALL_NOUNS[:60], PROMPT_TEMPLATES_30, layer)
    all_noun_mean = np.mean(list(noun_h.values()), axis=0)
    noun_fbase = {n: m - all_noun_mean for n, m in noun_h.items()}
    
    all_attrs = STIMULI["color_attrs"] + STIMULI["size_attrs"] + STIMULI["taste_attrs"]
    
    results = {}
    summary = {"top1_count": 0, "top3_count": 0, "top5_count": 0, "top10_count": 0, "top20_count": 0, "total": 0}
    
    for attr in all_attrs:
        direction, attr_tok_id = get_best_direction(model, tokenizer, device, attr, model_name, n_layers)
        if direction is None:
            continue
        
        attr_stats = {"top1": 0, "top3": 0, "top5": 0, "top10": 0, "top20": 0, "total": 0, "avg_rank": 0, "avg_logit_change": 0}
        
        for noun, h_noun in noun_h.items():
            f_base = noun_fbase.get(noun, np.zeros(D))
            
            logits_new, logits_diff = intervene_with_direction(
                model, device, h_noun, direction, layer, 
                f_base=f_base, alpha=alpha_fixed, beta=beta)
            
            rank_info = check_intervention_rank(logits_new, logits_diff, tokenizer, attr)
            
            attr_stats["top1"] += int(rank_info["top1"])
            attr_stats["top3"] += int(rank_info["top3"])
            attr_stats["top5"] += int(rank_info["top5"])
            attr_stats["top10"] += int(rank_info["top10"])
            attr_stats["top20"] += int(rank_info["top20"])
            attr_stats["total"] += 1
            attr_stats["avg_rank"] += rank_info["rank"]
            attr_stats["avg_logit_change"] += rank_info["logit_change"]
            
            # 前3个名词记录详细信息
            if attr_stats["total"] <= 3:
                L.log(f"    {attr}+{noun}: rank={rank_info['rank']}, top5={rank_info['top5_tokens']}, Δlogit={rank_info['logit_change']:.2f}")
        
        if attr_stats["total"] > 0:
            attr_stats["top1_rate"] = attr_stats["top1"] / attr_stats["total"] * 100
            attr_stats["top3_rate"] = attr_stats["top3"] / attr_stats["total"] * 100
            attr_stats["top5_rate"] = attr_stats["top5"] / attr_stats["total"] * 100
            attr_stats["top10_rate"] = attr_stats["top10"] / attr_stats["total"] * 100
            attr_stats["top20_rate"] = attr_stats["top20"] / attr_stats["total"] * 100
            attr_stats["avg_rank"] /= attr_stats["total"]
            attr_stats["avg_logit_change"] /= attr_stats["total"]
            
            summary["top1_count"] += attr_stats["top1"]
            summary["top3_count"] += attr_stats["top3"]
            summary["top5_count"] += attr_stats["top5"]
            summary["top10_count"] += attr_stats["top10"]
            summary["top20_count"] += attr_stats["top20"]
            summary["total"] += attr_stats["total"]
        
        results[attr] = attr_stats
        L.log(f"  {attr}: top1={attr_stats.get('top1_rate',0):.1f}%, top3={attr_stats.get('top3_rate',0):.1f}%, top5={attr_stats.get('top5_rate',0):.1f}%, top10={attr_stats.get('top10_rate',0):.1f}%, top20={attr_stats.get('top20_rate',0):.1f}%, avg_rank={attr_stats.get('avg_rank',0):.1f}")
    
    # 汇总
    if summary["total"] > 0:
        summary["top1_rate"] = summary["top1_count"] / summary["total"] * 100
        summary["top3_rate"] = summary["top3_count"] / summary["total"] * 100
        summary["top5_rate"] = summary["top5_count"] / summary["total"] * 100
        summary["top10_rate"] = summary["top10_count"] / summary["total"] * 100
        summary["top20_rate"] = summary["top20_count"] / summary["total"] * 100
        L.log(f"\n  ★汇总: top1={summary['top1_rate']:.1f}%, top3={summary['top3_rate']:.1f}%, top5={summary['top5_rate']:.1f}%, top10={summary['top10_rate']:.1f}%, top20={summary['top20_rate']:.1f}%")
    
    return {"details": results, "summary": summary}

# ===================== P340: 生成测试 =====================
def run_p340(model, tokenizer, device, n_layers, D, model_name):
    """干预后让模型生成文本, 检查是否包含属性词"""
    L.log("=== P340: 生成测试 ===")
    
    alpha_fixed = 0.5
    beta = get_best_beta(model_name)
    layer = get_best_layer(model_name, n_layers)
    
    # 生成用的模板
    gen_templates = [
        "The {word} is",
        "I saw a {word} that was",
        "This {word} looks very",
    ]
    
    test_nouns = ALL_NOUNS[:20]  # 20个名词
    all_attrs = STIMULI["color_attrs"] + STIMULI["size_attrs"] + STIMULI["taste_attrs"]
    
    # 只测关键属性(3类各4个)
    key_attrs = ["red","blue","black","white", "big","small","long","short", "sweet","fresh","soft","spicy"]
    
    results = {}
    total_gen = 0
    contain_attr = 0
    contain_related = 0
    
    for attr in key_attrs:
        direction, attr_tok_id = get_best_direction(model, tokenizer, device, attr, model_name, n_layers)
        if direction is None:
            continue
        
        attr_stats = {"total": 0, "contains_attr": 0, "contains_related": 0, "examples": []}
        
        for noun in test_nouns:
            for template in gen_templates:
                prompt = template.format(word=noun)
                toks = tokenizer(prompt, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    out = model(toks.input_ids, output_hidden_states=True)
                    h_noun = out.hidden_states[layer][0, -1].float().cpu().numpy()
                
                # 基线生成(不干预)
                with torch.no_grad():
                    base_gen_ids = model.generate(
                        toks.input_ids, max_new_tokens=20, 
                        do_sample=False, temperature=1.0,
                        pad_token_id=tokenizer.eos_token_id)
                    base_text = tokenizer.decode(base_gen_ids[0], skip_special_tokens=True)
                
                # 干预后生成
                # 需要用hook干预
                intervention_applied = [False]
                
                def make_hook(direction_vec, beta_val, f_base_vec, alpha_val):
                    def hook_fn(module, input, output):
                        if intervention_applied[0]:
                            return output
                        intervention_applied[0] = True
                        # output是tuple, 第一个是hidden_states
                        h = output[0][:, -1, :].detach().float().cpu().numpy()
                        h_new = h + alpha_val * f_base_vec + beta_val * direction_vec
                        h_new_tensor = torch.tensor(h_new, dtype=output[0].dtype, device=output[0].device)
                        new_output = (output[0].clone(),) + output[1:]
                        new_output[0][:, -1, :] = h_new_tensor
                        return new_output
                    return hook_fn
                
                # 计算f_base
                all_noun_h = collect_noun_hidden_states(model, tokenizer, device, [noun], PROMPT_TEMPLATES_30[:5], layer)
                all_noun_mean = np.mean(list(all_noun_h.values()), axis=0)
                f_base = all_noun_h[noun] - all_noun_mean
                
                # 注册hook
                target_layer = model.model.layers[layer] if hasattr(model, 'model') and hasattr(model.model, 'layers') else None
                if target_layer is None:
                    continue
                
                hook = target_layer.register_forward_hook(make_hook(direction, beta, f_base, alpha_fixed))
                
                try:
                    with torch.no_grad():
                        gen_ids = model.generate(
                            toks.input_ids, max_new_tokens=20,
                            do_sample=False, temperature=1.0,
                            pad_token_id=tokenizer.eos_token_id)
                        gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                except Exception as e:
                    gen_text = f"ERROR: {e}"
                finally:
                    hook.remove()
                
                # 检查生成文本
                attr_lower = attr.lower()
                gen_lower = gen_text.lower()
                base_lower = base_text.lower()
                
                contains = attr_lower in gen_lower
                # 相关词(如red→colorful, reddish等)
                related_words = [attr_lower, attr_lower+"ish", attr_lower+"er", attr_lower+"est", 
                                "un"+attr_lower, attr_lower+"ness", attr_lower+"ly"]
                contains_rel = any(w in gen_lower for w in related_words)
                
                attr_stats["total"] += 1
                if contains:
                    attr_stats["contains_attr"] += 1
                if contains_rel:
                    attr_stats["contains_related"] += 1
                
                total_gen += 1
                if contains:
                    contain_attr += 1
                if contains_rel:
                    contain_related += 1
                
                # 保存前2个例子
                if len(attr_stats["examples"]) < 2:
                    attr_stats["examples"].append({
                        "noun": noun, "attr": attr,
                        "base_gen": base_text[:80],
                        "intervened_gen": gen_text[:80],
                        "contains_attr": contains,
                    })
        
        if attr_stats["total"] > 0:
            attr_stats["attr_rate"] = attr_stats["contains_attr"] / attr_stats["total"] * 100
            attr_stats["related_rate"] = attr_stats["contains_related"] / attr_stats["total"] * 100
        
        results[attr] = attr_stats
        L.log(f"  {attr}: attr_in_gen={attr_stats.get('attr_rate',0):.1f}%, related_in_gen={attr_stats.get('related_rate',0):.1f}% ({attr_stats['total']} gens)")
    
    summary = {
        "total": total_gen,
        "contain_attr": contain_attr,
        "contain_related": contain_related,
        "attr_rate": contain_attr / total_gen * 100 if total_gen > 0 else 0,
        "related_rate": contain_related / total_gen * 100 if total_gen > 0 else 0,
    }
    L.log(f"\n  ★汇总: attr_in_gen={summary['attr_rate']:.1f}%, related_in_gen={summary['related_rate']:.1f}% ({total_gen} gens)")
    
    return {"details": results, "summary": summary}

# ===================== P341: 多属性组合干预 =====================
def run_p341(model, tokenizer, device, n_layers, D, model_name):
    """同时干预2个属性"""
    L.log("=== P341: 多属性组合干预 ===")
    
    alpha_fixed = 0.5
    beta = get_best_beta(model_name)
    layer = get_best_layer(model_name, n_layers)
    
    # 15个属性组合
    combos = [
        ("red", "big"), ("red", "sweet"), ("blue", "small"),
        ("green", "long"), ("white", "soft"), ("black", "heavy"),
        ("yellow", "fresh"), ("pink", "tiny"), ("orange", "crisp"),
        ("big", "sweet"), ("small", "sour"), ("long", "bitter"),
        ("short", "salty"), ("heavy", "spicy"), ("light", "mild"),
    ]
    
    test_nouns = ALL_NOUNS[:20]
    
    noun_h = collect_noun_hidden_states(model, tokenizer, device, test_nouns, PROMPT_TEMPLATES_30, layer)
    all_noun_mean = np.mean(list(noun_h.values()), axis=0)
    noun_fbase = {n: m - all_noun_mean for n, m in noun_h.items()}
    
    results = {}
    
    for attr1, attr2 in combos:
        dir1, id1 = get_best_direction(model, tokenizer, device, attr1, model_name, n_layers)
        dir2, id2 = get_best_direction(model, tokenizer, device, attr2, model_name, n_layers)
        if dir1 is None or dir2 is None:
            continue
        
        # 计算两方向夹角
        cos_angle = np.dot(dir1, dir2) / (np.linalg.norm(dir1) * np.linalg.norm(dir2) + 1e-10)
        
        # 3种组合策略
        for strategy in ["additive", "sequential", "weighted"]:
            combo_stats = {"attr1_top20": 0, "attr2_top20": 0, "both_top20": 0, "attr1_top5": 0, "attr2_top5": 0, "both_top5": 0, "total": 0}
            
            for noun, h_noun in noun_h.items():
                f_base = noun_fbase.get(noun, np.zeros(D))
                
                if strategy == "additive":
                    # 同时加两个方向
                    combo_dir = dir1 + dir2
                    h_new = h_noun + alpha_fixed * f_base + beta * combo_dir
                elif strategy == "sequential":
                    # 先加attr1再加attr2
                    h_mid = h_noun + alpha_fixed * f_base + beta * dir1
                    h_new = h_mid + beta * dir2
                else:  # weighted
                    # β/2每个方向
                    h_new = h_noun + alpha_fixed * f_base + (beta/2) * dir1 + (beta/2) * dir2
                
                with torch.no_grad():
                    h_tensor = torch.tensor(h_new, dtype=torch.bfloat16).unsqueeze(0).unsqueeze(0).to(device)
                    lm_head = model.lm_head if hasattr(model, 'lm_head') else model.get_output_embeddings()
                    norm = model.model.norm if hasattr(model, 'model') and hasattr(model.model, 'norm') else None
                    if norm is not None:
                        logits_new = lm_head(norm(h_tensor))[0, 0].float().cpu().numpy()
                    else:
                        logits_new = lm_head(h_tensor)[0, 0].float().cpu().numpy()
                
                # 检查两个属性
                rank1 = check_intervention_rank(logits_new, np.zeros_like(logits_new), tokenizer, attr1)
                rank2 = check_intervention_rank(logits_new, np.zeros_like(logits_new), tokenizer, attr2)
                
                combo_stats["total"] += 1
                if rank1["top20"]: combo_stats["attr1_top20"] += 1
                if rank2["top20"]: combo_stats["attr2_top20"] += 1
                if rank1["top20"] and rank2["top20"]: combo_stats["both_top20"] += 1
                if rank1["top5"]: combo_stats["attr1_top5"] += 1
                if rank2["top5"]: combo_stats["attr2_top5"] += 1
                if rank1["top5"] and rank2["top5"]: combo_stats["both_top5"] += 1
            
            if combo_stats["total"] > 0:
                combo_stats["attr1_top20_rate"] = combo_stats["attr1_top20"] / combo_stats["total"] * 100
                combo_stats["attr2_top20_rate"] = combo_stats["attr2_top20"] / combo_stats["total"] * 100
                combo_stats["both_top20_rate"] = combo_stats["both_top20"] / combo_stats["total"] * 100
                combo_stats["attr1_top5_rate"] = combo_stats["attr1_top5"] / combo_stats["total"] * 100
                combo_stats["attr2_top5_rate"] = combo_stats["attr2_top5"] / combo_stats["total"] * 100
                combo_stats["both_top5_rate"] = combo_stats["both_top5"] / combo_stats["total"] * 100
            
            key = f"{attr1}+{attr2}_{strategy}"
            results[key] = combo_stats
        
        L.log(f"  {attr1}+{attr2} (cos={cos_angle:.3f}): additive_both_top20={results[f'{attr1}+{attr2}_additive'].get('both_top20_rate',0):.1f}%, sequential={results[f'{attr1}+{attr2}_sequential'].get('both_top20_rate',0):.1f}%, weighted={results[f'{attr1}+{attr2}_weighted'].get('both_top20_rate',0):.1f}%")
    
    # 汇总: 每种策略的平均成功率
    strategy_summary = {}
    for strategy in ["additive", "sequential", "weighted"]:
        both_rates = [v.get("both_top20_rate", 0) for k, v in results.items() if strategy in k]
        both5_rates = [v.get("both_top5_rate", 0) for k, v in results.items() if strategy in k]
        strategy_summary[strategy] = {
            "avg_both_top20": np.mean(both_rates) if both_rates else 0,
            "avg_both_top5": np.mean(both5_rates) if both5_rates else 0,
        }
        L.log(f"  策略{strategy}: avg_both_top20={strategy_summary[strategy]['avg_both_top20']:.1f}%, avg_both_top5={strategy_summary[strategy]['avg_both_top5']:.1f}%")
    
    return {"details": results, "strategy_summary": strategy_summary}

# ===================== P342: 跨层稳定性测试 =====================
def run_p342(model, tokenizer, device, n_layers, D, model_name):
    """测试不同层的干预效果"""
    L.log("=== P342: 跨层稳定性测试 ===")
    
    alpha_fixed = 0.5
    beta = get_best_beta(model_name)
    layer_step = 2
    
    # 测试关键属性
    key_attrs = ["red","blue","white", "big","small","short", "sweet","fresh","soft"]
    
    test_nouns = ALL_NOUNS[:20]
    
    results = {}
    
    for attr in key_attrs:
        direction, attr_tok_id = get_best_direction(model, tokenizer, device, attr, model_name, n_layers)
        if direction is None:
            continue
        
        layer_results = {}
        
        for layer in range(0, n_layers, layer_step):
            # 收集该层的名词隐藏状态
            noun_h = collect_noun_hidden_states(model, tokenizer, device, test_nouns, PROMPT_TEMPLATES_30[:5], layer)
            all_noun_mean = np.mean(list(noun_h.values()), axis=0)
            noun_fbase = {n: m - all_noun_mean for n, m in noun_h.items()}
            
            top20_count = 0
            top5_count = 0
            total = 0
            
            for noun, h_noun in noun_h.items():
                f_base = noun_fbase.get(noun, np.zeros(D))
                
                logits_new, logits_diff = intervene_with_direction(
                    model, device, h_noun, direction, layer,
                    f_base=f_base, alpha=alpha_fixed, beta=beta)
                
                rank_info = check_intervention_rank(logits_new, logits_diff, tokenizer, attr)
                
                if rank_info["top20"]: top20_count += 1
                if rank_info["top5"]: top5_count += 1
                total += 1
            
            layer_results[f"L{layer}"] = {
                "top20_rate": top20_count / total * 100 if total > 0 else 0,
                "top5_rate": top5_count / total * 100 if total > 0 else 0,
                "total": total,
            }
        
        results[attr] = layer_results
        
        # 找出最优层和稳定层范围
        rates = [(l, v["top20_rate"]) for l, v in layer_results.items()]
        best_layer = max(rates, key=lambda x: x[1])
        stable_layers = [l for l, r in rates if r > 50]
        
        L.log(f"  {attr}: 最优层={best_layer[0]}({best_layer[1]:.1f}%), 稳定层(>50%)={len(stable_layers)}/{len(rates)}层")
    
    return results

# ===================== 主函数 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    model_name = args.model
    L.log(f"===== Phase LXI: 干预质量全面验证 — {model_name} =====")
    
    model, tokenizer, device = load_model(model_name)
    n_layers = len(model.model.layers) if hasattr(model, 'model') and hasattr(model.model, 'layers') else 24
    D = model.config.hidden_size
    
    L.log(f"  模型: {model_name}, 层数: {n_layers}, 维度: {D}, 设备: {device}")
    
    all_results = {"model": model_name, "n_layers": n_layers, "D": D}
    
    # P339: top-1/top-3/top-5/top-10精确排名
    p339 = run_p339(model, tokenizer, device, n_layers, D, model_name)
    all_results["p339"] = p339
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # P340: 生成测试
    p340 = run_p340(model, tokenizer, device, n_layers, D, model_name)
    all_results["p340"] = p340
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # P341: 多属性组合
    p341 = run_p341(model, tokenizer, device, n_layers, D, model_name)
    all_results["p341"] = p341
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # P342: 跨层稳定性
    p342 = run_p342(model, tokenizer, device, n_layers, D, model_name)
    all_results["p342"] = p342
    
    # 保存结果
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    out_path = OUT_DIR / f"phase_lxi_p339_342_{model_name}_{ts}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    L.log(f"\n结果已保存: {out_path}")
    
    # 释放模型
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    L.log(f"===== {model_name} 测试完成 =====")

if __name__ == "__main__":
    main()
