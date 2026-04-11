"""
Phase LXIV-P351/352/353/354: 语义干预的最终验证
======================================================================

Phase LXIII核心发现:
  1. attr_word位置干预=0%! last_token位置最有效!
  2. GLM4 red/gold/spicy=100%top20, avg_rank=1-2 (几乎top-1!)
  3. Qwen3生成测试last_token干预38-75%属性词出现率
  4. 信息注入理论: 干预空白位置>>增强已有位置

Phase LXIV核心目标: 语义干预的最终验证

四大实验:
  P351: Top-1精确验证
    - GLM4 last_token干预后, top-5具体是什么词?
    - 干预方向是否只让目标词升到top-1, 还是改变了整个分布?
    - 36属性 × 60名词 × 5β = 10800干预/模型

  P352: 矛盾信号测试 ★★★核心★★★
    - 在"The fire is"中注入"cold"方向 → "The cold fire is"?
    - 在"The ice is"中注入"hot"方向 → "The hot ice is"?
    - 测试信息注入是否能战胜常识记忆
    - 30个矛盾组合 × 20名词 × 5β = 3000干预/模型

  P353: 语义一致性验证
    - 干预后的top-10是否都是语义相关词?
    - 如注入"red"后, top-10是否包含red/crimson/scarlet/ruby?
    - 36属性 × 20名词 × 3β = 2160干预/模型

  P354: β精确调控
    - 找到让属性词成为top-1的最小β
    - β从0.1到20, 每步0.1-1.0
    - 12属性 × 20名词 × 20β = 4800干预/模型

数据规模: ~20000干预/模型
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

LOG_FILE = OUT_DIR / "phase_lxiv_log.txt"

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

ALL_ATTRS = STIMULI["color_attrs"] + STIMULI["taste_attrs"] + STIMULI["size_attrs"]

# 矛盾信号组合: (名词, 常识属性, 矛盾属性)
CONTRADICTION_PAIRS = [
    # 热vs冷
    ("fire", "hot", "cold"),
    ("ice", "cold", "hot"),
    ("snow", "cold", "hot"),
    ("sun", "hot", "cold"),
    ("lava", "hot", "cold"),
    ("furnace", "hot", "cold"),
    # 甜vs苦/酸
    ("honey", "sweet", "bitter"),
    ("sugar", "sweet", "sour"),
    ("candy", "sweet", "salty"),
    ("lemon", "sour", "sweet"),
    ("vinegar", "sour", "sweet"),
    ("coffee", "bitter", "sweet"),
    # 大vs小
    ("elephant", "big", "tiny"),
    ("mountain", "big", "small"),
    ("ocean", "big", "tiny"),
    ("ant", "small", "huge"),
    ("grain", "tiny", "big"),
    ("pebble", "small", "huge"),
    # 颜色矛盾
    ("grass", "green", "red"),
    ("sky", "blue", "black"),
    ("blood", "red", "white"),
    ("night", "black", "white"),
    ("snow", "white", "black"),
    ("gold", "gold", "blue"),
    # 硬vs软
    ("rock", "hard", "soft"),
    ("steel", "hard", "soft"),
    ("pillow", "soft", "hard"),
    ("sponge", "soft", "hard"),
    ("feather", "soft", "hard"),
    ("diamond", "hard", "soft"),
    # 快vs慢
    ("cheetah", "fast", "slow"),
    ("turtle", "slow", "fast"),
    ("rocket", "fast", "slow"),
    ("snail", "slow", "fast"),
]

# 语义相关词映射 (用于P353)
SEMANTIC_GROUPS = {
    "red": ["red", "crimson", "scarlet", "ruby", "maroon", "cherry", "vermilion", "rose"],
    "green": ["green", "emerald", "lime", "olive", "jade", "mint", "forest", "verdant"],
    "blue": ["blue", "azure", "navy", "cobalt", "sapphire", "indigo", "cerulean", "cyan"],
    "yellow": ["yellow", "golden", "amber", "lemon", "canary", "saffron", "blond"],
    "white": ["white", "snowy", "pale", "ivory", "pearl", "alabaster", "milky"],
    "black": ["black", "dark", "ebony", "jet", "onyx", "midnight", "raven"],
    "sweet": ["sweet", "sugary", "honeyed", "syrupy", "candied", "fruity", "delicious"],
    "sour": ["sour", "tart", "acidic", "tangy", "bitter", "sharp", "vinegary"],
    "hot": ["hot", "warm", "fiery", "burning", "scorching", "heated", "blazing"],
    "cold": ["cold", "cool", "chilly", "freezing", "icy", "frigid", "frosty"],
    "big": ["big", "large", "huge", "enormous", "giant", "massive", "vast", "immense"],
    "small": ["small", "tiny", "little", "miniature", "petite", "microscopic", "minute"],
    "hard": ["hard", "solid", "firm", "rigid", "stiff", "tough", "dense"],
    "soft": ["soft", "gentle", "tender", "yielding", "fluffy", "cushy", "plush"],
    "fast": ["fast", "quick", "rapid", "swift", "speedy", "hasty", "fleet"],
    "slow": ["slow", "sluggish", "leisurely", "gradual", "unhurried", "languid"],
}

def get_attr_direction(model, tokenizer, attr, method="lm_head"):
    """获取属性词的方向向量"""
    attr_tok_ids = tokenizer.encode(attr, add_special_tokens=False)
    if len(attr_tok_ids) == 0:
        return None
    
    if method == "lm_head":
        lm_head = model.lm_head if hasattr(model, 'lm_head') else model.get_output_embeddings()
        direction = lm_head.weight[attr_tok_ids[0]].detach().cpu().float().numpy().copy()
    elif method == "embedding":
        embed_layer = model.get_input_embeddings()
        direction = embed_layer.weight[attr_tok_ids[0]].detach().cpu().float().numpy().copy()
    else:
        return None
    
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm
    
    return direction

def intervene_at_position_and_get_logits(model, tokenizer, device, prompt, direction, beta, 
                                          target_positions=None, alpha=1.0):
    """在指定token位置干预embedding, 然后通过整个模型获取logits"""
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    seq_len = input_ids.shape[1]
    
    with torch.no_grad():
        embed_layer = model.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids).clone()
        
        direction_tensor = torch.tensor(direction, dtype=inputs_embeds.dtype, device=device)
        
        if target_positions is None:
            inputs_embeds[0, -1, :] = inputs_embeds[0, -1, :] + alpha * beta * direction_tensor
        else:
            for pos in target_positions:
                if 0 <= pos < seq_len:
                    inputs_embeds[0, pos, :] = inputs_embeds[0, pos, :] + alpha * beta * direction_tensor
        
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        out = model(inputs_embeds=inputs_embeds, position_ids=position_ids)
        logits = out.logits[0, -1].float().cpu().numpy()
    
    return logits

def get_top_k_words(logits, tokenizer, k=10):
    """获取logits中的top-k词"""
    sorted_ids = np.argsort(logits)[::-1][:k]
    words = []
    for tid in sorted_ids:
        word = tokenizer.decode([tid]).strip()
        prob = float(np.exp(logits[tid] - np.max(logits)))  # 简化的softmax
        words.append({"word": word, "token_id": int(tid), "logit": float(logits[tid])})
    return words

def check_rank_in_logits(logits, tokenizer, target_word, top_k=20):
    """检查目标词在logits中的排名"""
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

# ===================== P351: Top-1精确验证 =====================
def run_p351(model, tokenizer, device, model_name):
    """验证GLM4等模型的top-1具体是什么词"""
    L.log("=== P351: Top-1精确验证 ===")
    
    method = "embedding" if model_name == "qwen3" else "lm_head"
    beta_range = [1.0, 3.0, 5.0, 8.0]
    
    results = {}
    total_attrs = len(ALL_ATTRS)
    
    for ai, attr in enumerate(ALL_ATTRS):
        L.log(f"  [{ai+1}/{total_attrs}] {attr}")
        direction = get_attr_direction(model, tokenizer, attr, method)
        if direction is None:
            continue
        
        attr_results = {}
        test_nouns = ALL_NOUNS[:60]
        
        for beta in beta_range:
            stats = {"top1": 0, "top3": 0, "top5": 0, "top10": 0, "top20": 0, 
                     "total": 0, "avg_rank": 0, "top5_words": {}}
            examples = []
            
            for noun in test_nouns:
                prompt = f"The {noun} is"
                
                # 基线logits
                toks = tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    out_base = model(toks.input_ids)
                    logits_base = out_base.logits[0, -1].float().cpu().numpy()
                
                # 干预logits
                try:
                    logits_intervened = intervene_at_position_and_get_logits(
                        model, tokenizer, device, prompt, direction, beta, None)
                except:
                    continue
                
                rank_info = check_rank_in_logits(logits_intervened, tokenizer, attr)
                
                if rank_info["hit"]:
                    stats["total"] += 1
                    stats["avg_rank"] += rank_info["rank"]
                    if rank_info["top1"]: stats["top1"] += 1
                    if rank_info["top3"]: stats["top3"] += 1
                    if rank_info["top5"]: stats["top5"] += 1
                    if rank_info["top10"]: stats["top10"] += 1
                    if rank_info["top20"]: stats["top20"] += 1
                    
                    # 记录top-5词
                    top5 = get_top_k_words(logits_intervened, tokenizer, 5)
                    for item in top5:
                        w = item["word"]
                        stats["top5_words"][w] = stats["top5_words"].get(w, 0) + 1
                    
                    # 记录基线top-1 vs 干预top-1
                    base_top1 = tokenizer.decode([np.argsort(logits_base)[::-1][0]]).strip()
                    int_top1 = top5[0]["word"]
                    
                    if len(examples) < 5:
                        examples.append({
                            "noun": noun, "beta": beta,
                            "rank": rank_info["rank"],
                            "base_top1": base_top1,
                            "intervened_top5": [item["word"] for item in top5],
                        })
            
            if stats["total"] > 0:
                key = f"b{beta}"
                stats["top1%"] = round(stats["top1"] / stats["total"] * 100, 1)
                stats["top3%"] = round(stats["top3"] / stats["total"] * 100, 1)
                stats["top5%"] = round(stats["top5"] / stats["total"] * 100, 1)
                stats["top10%"] = round(stats["top10"] / stats["total"] * 100, 1)
                stats["top20%"] = round(stats["top20"] / stats["total"] * 100, 1)
                stats["avg_rank"] = round(stats["avg_rank"] / stats["total"], 1)
                # top5词频率排序
                stats["top5_words_ranked"] = sorted(stats["top5_words"].items(), key=lambda x: -x[1])[:10]
                stats["examples"] = examples
                attr_results[key] = stats
        
        # 报告
        best_beta = max(attr_results.keys(), key=lambda k: attr_results[k].get("top20%", 0)) if attr_results else None
        if best_beta:
            r = attr_results[best_beta]
            L.log(f"    β={best_beta}: top1%={r['top1%']}, top20%={r['top20%']}, "
                  f"avg_rank={r['avg_rank']}, top5_words={r.get('top5_words_ranked',[][:3])}")
        
        results[attr] = attr_results
    
    return results

# ===================== P352: 矛盾信号测试 =====================
def run_p352(model, tokenizer, device, model_name):
    """矛盾信号测试: 注入与常识矛盾的属性"""
    L.log("=== P352: 矛盾信号测试 ===")
    
    method = "embedding" if model_name == "qwen3" else "lm_head"
    beta_range = [1.0, 3.0, 5.0, 8.0, 12.0]
    
    results = {}
    total_pairs = len(CONTRADICTION_PAIRS)
    
    for pi, (noun, common_attr, contra_attr) in enumerate(CONTRADICTION_PAIRS):
        L.log(f"  [{pi+1}/{total_pairs}] {noun}: common={common_attr}, contra={contra_attr}")
        
        # 获取矛盾属性的方向
        contra_direction = get_attr_direction(model, tokenizer, contra_attr, method)
        if contra_direction is None:
            continue
        
        # 获取常识属性的方向(作为对照)
        common_direction = get_attr_direction(model, tokenizer, common_attr, method)
        
        pair_results = {}
        
        for beta in beta_range:
            stats = {
                "contra_top1": 0, "contra_top5": 0, "contra_top10": 0, "contra_top20": 0,
                "common_top1": 0, "common_top5": 0, "common_top10": 0, "common_top20": 0,
                "total": 0,
                "contra_avg_rank": 0, "common_avg_rank": 0,
                "examples": []
            }
            
            # 测试3个模板
            for template in ["The {word} is", "A {word} can be", "This {word} has"]:
                prompt = template.format(word=noun)
                
                # 基线: 不干预
                toks = tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    out_base = model(toks.input_ids)
                    logits_base = out_base.logits[0, -1].float().cpu().numpy()
                
                # 注入矛盾属性方向
                try:
                    logits_contra = intervene_at_position_and_get_logits(
                        model, tokenizer, device, prompt, contra_direction, beta, None)
                except:
                    continue
                
                # 检查矛盾属性和常识属性的排名
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
                
                # 记录示例
                if len(stats["examples"]) < 2:
                    base_top5 = get_top_k_words(logits_base, tokenizer, 5)
                    contra_top5 = get_top_k_words(logits_contra, tokenizer, 5)
                    stats["examples"].append({
                        "prompt": prompt,
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
                
                pair_results[key] = stats
        
        # 报告最佳结果
        best_beta = max(pair_results.keys(), key=lambda k: pair_results[k].get("contra_top20%", 0)) if pair_results else None
        if best_beta:
            r = pair_results[best_beta]
            L.log(f"    β={best_beta}: contra_top20%={r['contra_top20%']}(rank={r['contra_avg_rank']}), "
                  f"common_top20%={r['common_top20%']}(rank={r['common_avg_rank']})")
        
        results[f"{noun}_{common_attr}_vs_{contra_attr}"] = pair_results
    
    return results

# ===================== P353: 语义一致性验证 =====================
def run_p353(model, tokenizer, device, model_name):
    """验证干预后的top-10是否语义一致"""
    L.log("=== P353: 语义一致性验证 ===")
    
    method = "embedding" if model_name == "qwen3" else "lm_head"
    beta_range = [3.0, 5.0, 8.0]
    
    results = {}
    test_attrs = list(SEMANTIC_GROUPS.keys())
    total_attrs = len(test_attrs)
    
    for ai, attr in enumerate(test_attrs):
        L.log(f"  [{ai+1}/{total_attrs}] {attr}")
        direction = get_attr_direction(model, tokenizer, attr, method)
        if direction is None:
            continue
        
        semantic_group = SEMANTIC_GROUPS[attr]
        attr_results = {}
        test_nouns = ALL_NOUNS[:20]
        
        for beta in beta_range:
            stats = {
                "total": 0, "semantic_hits": 0, "semantic_rate": 0,
                "top10_words": {}, "top10_semantic": {}
            }
            
            for noun in test_nouns:
                prompt = f"The {noun} is"
                
                try:
                    logits = intervene_at_position_and_get_logits(
                        model, tokenizer, device, prompt, direction, beta, None)
                except:
                    continue
                
                top10 = get_top_k_words(logits, tokenizer, 10)
                
                stats["total"] += 1
                semantic_count = 0
                
                for item in top10:
                    w = item["word"].lower()
                    stats["top10_words"][w] = stats["top10_words"].get(w, 0) + 1
                    
                    # 检查是否语义相关
                    is_semantic = any(s in w or w in s for s in semantic_group)
                    if is_semantic:
                        semantic_count += 1
                        stats["top10_semantic"][w] = stats["top10_semantic"].get(w, 0) + 1
                
                stats["semantic_hits"] += semantic_count
            
            if stats["total"] > 0:
                key = f"b{beta}"
                stats["semantic_rate"] = round(stats["semantic_hits"] / (stats["total"] * 10) * 100, 1)
                stats["top10_words_ranked"] = sorted(stats["top10_words"].items(), key=lambda x: -x[1])[:15]
                stats["top10_semantic_ranked"] = sorted(stats["top10_semantic"].items(), key=lambda x: -x[1])[:10]
                attr_results[key] = stats
        
        # 报告
        best_beta = max(attr_results.keys(), key=lambda k: attr_results[k].get("semantic_rate", 0)) if attr_results else None
        if best_beta:
            r = attr_results[best_beta]
            L.log(f"    β={best_beta}: semantic_rate={r['semantic_rate']}%, "
                  f"top_words={r['top10_words_ranked'][:5]}, "
                  f"semantic_words={r['top10_semantic_ranked'][:5]}")
        
        results[attr] = attr_results
    
    return results

# ===================== P354: β精确调控 =====================
def run_p354(model, tokenizer, device, model_name):
    """找到让属性词成为top-1的最小β"""
    L.log("=== P354: β精确调控 ===")
    
    method = "embedding" if model_name == "qwen3" else "lm_head"
    # 细粒度β范围
    beta_range = [0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0]
    
    results = {}
    # 只测试部分属性(最重要的)
    test_attrs = ["red", "green", "blue", "sweet", "sour", "hot", "cold", "big", "small", "soft", "hard", "fast"]
    total_attrs = len(test_attrs)
    
    for ai, attr in enumerate(test_attrs):
        L.log(f"  [{ai+1}/{total_attrs}] {attr}")
        direction = get_attr_direction(model, tokenizer, attr, method)
        if direction is None:
            continue
        
        attr_results = {}
        test_nouns = ALL_NOUNS[:20]
        
        for beta in beta_range:
            stats = {"top1": 0, "top5": 0, "top20": 0, "total": 0, "avg_rank": 0}
            
            for noun in test_nouns:
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
                attr_results[key] = stats
        
        # 找到top-1的最小β
        min_beta_top1 = None
        min_beta_top5 = None
        min_beta_top20 = None
        
        for beta in beta_range:
            key = f"b{beta}"
            if key in attr_results:
                if attr_results[key]["top1"] > 0 and min_beta_top1 is None:
                    min_beta_top1 = beta
                if attr_results[key]["top5"] > 0 and min_beta_top5 is None:
                    min_beta_top5 = beta
                if attr_results[key]["top20"] > 0 and min_beta_top20 is None:
                    min_beta_top20 = beta
        
        if min_beta_top1 or min_beta_top5 or min_beta_top20:
            L.log(f"    min_β(top1)={min_beta_top1}, min_β(top5)={min_beta_top5}, min_β(top20)={min_beta_top20}")
        
        results[attr] = {
            "beta_sweep": attr_results,
            "min_beta_top1": min_beta_top1,
            "min_beta_top5": min_beta_top5,
            "min_beta_top20": min_beta_top20,
        }
    
    return results

# ===================== 主函数 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    model_name = args.model
    L.log(f"===== Phase LXIV: 语义干预的最终验证 =====")
    L.log(f"模型: {model_name}")
    
    # 加载模型
    L.log("加载模型...")
    model, tokenizer, device = load_model(model_name)
    n_layers = len(model.model.layers) if hasattr(model, 'model') and hasattr(model.model, 'layers') else 0
    D = model.config.hidden_size
    
    L.log(f"模型: {model_name}, 层数: {n_layers}, 维度: {D}, 设备: {device}")
    
    # ========== P351: Top-1精确验证 ==========
    L.log("P351: Top-1精确验证...")
    p351_results = run_p351(model, tokenizer, device, model_name)
    
    # ========== P352: 矛盾信号测试 ==========
    L.log("P352: 矛盾信号测试...")
    p352_results = run_p352(model, tokenizer, device, model_name)
    
    # ========== P353: 语义一致性验证 ==========
    L.log("P353: 语义一致性验证...")
    p353_results = run_p353(model, tokenizer, device, model_name)
    
    # ========== P354: β精确调控 ==========
    L.log("P354: β精确调控...")
    p354_results = run_p354(model, tokenizer, device, model_name)
    
    # ========== 保存结果 ==========
    all_results = {
        "model": model_name,
        "n_layers": n_layers,
        "D": D,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "p351_top1_verification": p351_results,
        "p352_contradiction_test": p352_results,
        "p353_semantic_consistency": p353_results,
        "p354_beta_precision": p354_results,
    }
    
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = OUT_DIR / f"phase_lxiv_p351_354_{model_name}_{ts}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    L.log(f"结果已保存: {out_path}")
    
    # ========== 汇总 ==========
    L.log("\n===== 汇总 =====")
    
    # P351汇总
    L.log("\nP351: Top-1精确验证")
    for attr in ALL_ATTRS[:12]:
        if attr in p351_results:
            attr_res = p351_results[attr]
            best_beta = max(attr_res.keys(), key=lambda k: attr_res[k].get("top20%", 0)) if attr_res else None
            if best_beta:
                r = attr_res[best_beta]
                L.log(f"  {attr}(β={best_beta}): top1%={r['top1%']}, top20%={r['top20%']}, "
                      f"avg_rank={r['avg_rank']}, top5={r.get('top5_words_ranked',[][:3])}")
    
    # P352汇总
    L.log("\nP352: 矛盾信号测试")
    for key in list(p352_results.keys())[:10]:
        pair_res = p352_results[key]
        best_beta = max(pair_res.keys(), key=lambda k: pair_res[k].get("contra_top20%", 0)) if pair_res else None
        if best_beta:
            r = pair_res[best_beta]
            L.log(f"  {key}(β={best_beta}): contra_top20%={r['contra_top20%']}, "
                  f"common_top20%={r['common_top20%']}")
    
    # P353汇总
    L.log("\nP353: 语义一致性")
    for attr in list(p353_results.keys())[:10]:
        attr_res = p353_results[attr]
        best_beta = max(attr_res.keys(), key=lambda k: attr_res[k].get("semantic_rate", 0)) if attr_res else None
        if best_beta:
            r = attr_res[best_beta]
            L.log(f"  {attr}(β={best_beta}): semantic_rate={r['semantic_rate']}%")
    
    # P354汇总
    L.log("\nP354: β精确调控")
    p354_test_attrs = ["red", "green", "blue", "sweet", "sour", "hot", "cold", "big", "small", "soft", "hard", "fast"]
    for attr in p354_test_attrs:
        if attr in p354_results:
            r = p354_results[attr]
            L.log(f"  {attr}: min_β(top1)={r['min_beta_top1']}, "
                  f"min_β(top5)={r['min_beta_top5']}, "
                  f"min_β(top20)={r['min_beta_top20']}")
    
    # 释放GPU
    del model
    gc.collect()
    torch.cuda.empty_cache()
    L.log("GPU已释放")
    
    L.close()

if __name__ == "__main__":
    main()
