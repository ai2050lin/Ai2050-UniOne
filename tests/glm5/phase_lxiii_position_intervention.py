"""
Phase LXIII-P347/348/349/350: 位置感知语义干预
======================================================================

Phase LXII核心发现:
  1. L0干预改善排名6x(Qwen3), 但不达top-1
  2. GLM4唯一有top-1成功: sour=15%, thick/red=8.3%
  3. 核心问题: 修改位置不对! 在"is"上加"red"方向≠在"red"位置加方向
  4. 语义编码是位置相关的!

Phase LXIII核心假设: 在属性词位置加方向 = 真正的语义干预

四大实验:
  P347: 不同位置干预对比
    - 位置1: 在属性词token位置加方向 (如"red" in "The red apple is")
    - 位置2: 在名词token位置加方向 (如"apple" in "The apple is")
    - 位置3: 在最后token位置加方向 (如"is" in "The apple is")
    - 对比三种位置的干预效果
    - 36属性 × 60名词 × 3位置 × 5β = 32400干预/模型

  P348: 属性词位置精确干预
    - 在"attr noun"中, 只在属性词位置加lm_head方向
    - 与在"noun"中加方向的L0干预对比
    - 36属性 × 60名词 × 5β = 10800干预/模型

  P349: 多token属性的位置策略
    - 某些属性词可能被分词为多个token
    - 在第一个token/最后一个token/所有token上分别加方向
    - 12属性 × 60名词 × 3策略 × 5β = 10800干预/模型

  P350: 生成测试 - 位置干预后的生成质量
    - 在属性词位置干预后让模型生成
    - 检查生成文本是否包含目标属性
    - 18属性 × 20名词 × 3模板 = 1080生成/模型

数据规模: 36属性 × 60名词 × 3位置 × 5β × 3模型 = 97200干预
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

LOG_FILE = OUT_DIR / "phase_lxiii_log.txt"

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

ALL_ATTRS = STIMULI["color_attrs"] + STIMULI["taste_attrs"] + STIMULI["size_attrs"]

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
    """在指定token位置干预embedding, 然后通过整个模型获取logits
    
    target_positions: list of int, 要干预的token位置索引
    如果为None, 干预最后一个token
    """
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    seq_len = input_ids.shape[1]
    
    with torch.no_grad():
        # 1. 获取embedding
        embed_layer = model.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids).clone()  # (1, seq_len, D)
        
        # 2. 在指定位置添加方向
        direction_tensor = torch.tensor(direction, dtype=inputs_embeds.dtype, device=device)
        
        if target_positions is None:
            # 干预最后一个token
            inputs_embeds[0, -1, :] = inputs_embeds[0, -1, :] + alpha * beta * direction_tensor
        else:
            # 干预指定位置
            for pos in target_positions:
                if 0 <= pos < seq_len:
                    inputs_embeds[0, pos, :] = inputs_embeds[0, pos, :] + alpha * beta * direction_tensor
        
        # 3. Forward
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        out = model(inputs_embeds=inputs_embeds, position_ids=position_ids)
        logits = out.logits[0, -1].float().cpu().numpy()
    
    return logits

def intervene_at_position_and_generate(model, tokenizer, device, prompt, direction, beta,
                                        target_positions=None, alpha=1.0, max_new_tokens=25):
    """在指定token位置干预后让模型生成文本"""
    intervention_applied = [False]
    direction_tensor = torch.tensor(direction, dtype=torch.bfloat16, device=device)
    
    # 预先计算位置
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    seq_len = toks.input_ids.shape[1]
    if target_positions is None:
        target_positions = [seq_len - 1]
    
    # 用embedding hook方式, 只在第一次forward时干预
    call_count = [0]
    def embedding_hook(module, input, output):
        call_count[0] += 1
        if call_count[0] > 1:
            return output  # 后续forward不再干预
        
        if isinstance(output, tuple):
            embeds = output[0].clone()
        else:
            embeds = output.clone()
        
        for pos in target_positions:
            if 0 <= pos < embeds.shape[1]:
                embeds[0, pos, :] = embeds[0, pos, :] + alpha * beta * direction_tensor
        
        if isinstance(output, tuple):
            return (embeds,) + output[1:]
        return embeds
    
    embed_layer = model.get_input_embeddings()
    hook = embed_layer.register_forward_hook(embedding_hook)
    
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

def check_rank_in_logits(logits, tokenizer, target_word, top_k=20):
    """检查目标词在logits中的排名"""
    attr_tok_ids = tokenizer.encode(target_word, add_special_tokens=False)
    if len(attr_tok_ids) == 0:
        return {"hit": False, "rank": -1, "top_k": False, "top1": False, "top3": False, "top5": False, "top10": False, "top20": False}
    
    attr_tok_id = attr_tok_ids[0]
    sorted_ids = np.argsort(logits)[::-1]
    rank = int(np.where(sorted_ids == attr_tok_id)[0][0]) + 1 if attr_tok_id in sorted_ids else len(logits)
    
    return {
        "hit": True,
        "rank": rank,
        "top_k": rank <= top_k,
        "top1": rank <= 1,
        "top3": rank <= 3,
        "top5": rank <= 5,
        "top10": rank <= 10,
        "top20": rank <= 20,
    }

def find_token_positions(tokenizer, prompt, target_word):
    """找到目标词在prompt中的token位置"""
    toks = tokenizer(prompt, return_tensors="pt")
    input_ids = toks.input_ids[0].tolist()
    
    target_tok_ids = tokenizer.encode(target_word, add_special_tokens=False)
    if len(target_tok_ids) == 0:
        return []
    
    positions = []
    for i in range(len(input_ids) - len(target_tok_ids) + 1):
        if input_ids[i:i+len(target_tok_ids)] == target_tok_ids:
            positions = list(range(i, i + len(target_tok_ids)))
            break
    
    return positions

# ===================== P347: 不同位置干预对比 =====================
def run_p347(model, tokenizer, device, model_name):
    """对比在属性词位置/名词位置/最后token位置干预的效果"""
    L.log("=== P347: 不同位置干预对比 ===")
    
    # 选择方法: Qwen3用embedding, GLM4/DS7B用lm_head
    method = "embedding" if model_name == "qwen3" else "lm_head"
    beta_range = [1.0, 2.0, 3.0, 5.0, 8.0]
    
    results = {}
    total_attrs = len(ALL_ATTRS)
    
    for ai, attr in enumerate(ALL_ATTRS):
        L.log(f"  [{ai+1}/{total_attrs}] {attr}")
        direction = get_attr_direction(model, tokenizer, attr, method)
        if direction is None:
            continue
        
        attr_results = {}
        test_nouns = ALL_NOUNS[:60]
        
        for position_type in ["attr_word", "noun_word", "last_token"]:
            for beta in beta_range:
                stats = {"top1": 0, "top3": 0, "top5": 0, "top10": 0, "top20": 0, "total": 0, "avg_rank": 0}
                examples = []
                
                for noun in test_nouns:
                    # 三种位置的prompt
                    if position_type == "attr_word":
                        # "The red apple is" - 在"red"位置加方向
                        prompt = f"The {attr} {noun} is"
                        target_positions = find_token_positions(tokenizer, prompt, attr)
                        if len(target_positions) == 0:
                            continue
                    elif position_type == "noun_word":
                        # "The apple is" - 在"apple"位置加方向
                        prompt = f"The {noun} is"
                        target_positions = find_token_positions(tokenizer, prompt, noun)
                        if len(target_positions) == 0:
                            continue
                    else:  # last_token
                        # "The apple is" - 在"is"位置加方向
                        prompt = f"The {noun} is"
                        target_positions = None  # 最后一个token
                    
                    try:
                        logits = intervene_at_position_and_get_logits(
                            model, tokenizer, device, prompt, direction, beta, target_positions)
                    except Exception as e:
                        continue
                    
                    rank_info = check_rank_in_logits(logits, tokenizer, attr)
                    
                    if rank_info["hit"]:
                        stats["total"] += 1
                        stats["avg_rank"] += rank_info["rank"]
                        if rank_info["top1"]: stats["top1"] += 1
                        if rank_info["top3"]: stats["top3"] += 1
                        if rank_info["top5"]: stats["top5"] += 1
                        if rank_info["top10"]: stats["top10"] += 1
                        if rank_info["top20"]: stats["top20"] += 1
                        
                        if len(examples) < 3:
                            top5_ids = np.argsort(logits)[::-1][:5]
                            top5_words = [tokenizer.decode([tid]) for tid in top5_ids]
                            examples.append({
                                "noun": noun, "position_type": position_type,
                                "rank": rank_info["rank"],
                                "top5": top5_words
                            })
                
                if stats["total"] > 0:
                    key = f"{position_type}_b{beta}"
                    stats["top1%"] = round(stats["top1"] / stats["total"] * 100, 1)
                    stats["top3%"] = round(stats["top3"] / stats["total"] * 100, 1)
                    stats["top5%"] = round(stats["top5"] / stats["total"] * 100, 1)
                    stats["top10%"] = round(stats["top10"] / stats["total"] * 100, 1)
                    stats["top20%"] = round(stats["top20"] / stats["total"] * 100, 1)
                    stats["avg_rank"] = round(stats["avg_rank"] / stats["total"], 1)
                    stats["examples"] = examples
                    attr_results[key] = stats
            
            # 报告最佳beta
            best_key = max(attr_results.keys(), key=lambda k: attr_results[k].get("top20%", 0)) if attr_results else None
            if best_key:
                L.log(f"    {position_type}: best={best_key}, top20%={attr_results[best_key]['top20%']}%, avg_rank={attr_results[best_key]['avg_rank']}")
        
        results[attr] = attr_results
    
    return results

# ===================== P348: 属性词位置精确干预 =====================
def run_p348(model, tokenizer, device, model_name):
    """在属性词位置精确干预, 与L0干预对比"""
    L.log("=== P348: 属性词位置精确干预 ===")
    
    method = "embedding" if model_name == "qwen3" else "lm_head"
    beta_range = [1.0, 2.0, 3.0, 5.0, 8.0]
    
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
            stats_pos = {"top1": 0, "top3": 0, "top5": 0, "top10": 0, "top20": 0, "total": 0, "avg_rank": 0}
            stats_L0 = {"top1": 0, "top3": 0, "top5": 0, "top10": 0, "top20": 0, "total": 0, "avg_rank": 0}
            stats_direct = {"top1": 0, "top3": 0, "top5": 0, "top10": 0, "top20": 0, "total": 0, "avg_rank": 0}
            
            for noun in test_nouns:
                # 方式1: 在属性词位置干预 (prompt包含attr)
                prompt_with_attr = f"The {attr} {noun} is"
                attr_positions = find_token_positions(tokenizer, prompt_with_attr, attr)
                if len(attr_positions) == 0:
                    continue
                
                try:
                    logits_pos = intervene_at_position_and_get_logits(
                        model, tokenizer, device, prompt_with_attr, direction, beta, attr_positions)
                except:
                    continue
                
                rank_pos = check_rank_in_logits(logits_pos, tokenizer, attr)
                
                # 方式2: L0干预 (prompt不含attr, 在最后token干预)
                prompt_no_attr = f"The {noun} is"
                try:
                    logits_L0 = intervene_at_position_and_get_logits(
                        model, tokenizer, device, prompt_no_attr, direction, beta, None)
                except:
                    continue
                
                rank_L0 = check_rank_in_logits(logits_L0, tokenizer, attr)
                
                # 方式3: 直接输入(attr+noun, 不干预)
                toks = tokenizer(prompt_with_attr, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model(toks.input_ids)
                    logits_direct = out.logits[0, -1].float().cpu().numpy()
                rank_direct = check_rank_in_logits(logits_direct, tokenizer, attr)
                
                # 统计
                for stats, rank_info in [(stats_pos, rank_pos), (stats_L0, rank_L0), (stats_direct, rank_direct)]:
                    if rank_info["hit"]:
                        stats["total"] += 1
                        stats["avg_rank"] += rank_info["rank"]
                        if rank_info["top1"]: stats["top1"] += 1
                        if rank_info["top3"]: stats["top3"] += 1
                        if rank_info["top5"]: stats["top5"] += 1
                        if rank_info["top10"]: stats["top10"] += 1
                        if rank_info["top20"]: stats["top20"] += 1
            
            # 计算百分比
            for label, stats in [("attr_pos", stats_pos), ("L0_last", stats_L0), ("direct", stats_direct)]:
                if stats["total"] > 0:
                    stats["top1%"] = round(stats["top1"] / stats["total"] * 100, 1)
                    stats["top3%"] = round(stats["top3"] / stats["total"] * 100, 1)
                    stats["top5%"] = round(stats["top5"] / stats["total"] * 100, 1)
                    stats["top10%"] = round(stats["top10"] / stats["total"] * 100, 1)
                    stats["top20%"] = round(stats["top20"] / stats["total"] * 100, 1)
                    stats["avg_rank"] = round(stats["avg_rank"] / stats["total"], 1)
            
            key = f"b{beta}"
            attr_results[key] = {
                "attr_pos": stats_pos,
                "L0_last": stats_L0,
                "direct": stats_direct,
            }
        
        # 报告最佳结果
        best_beta = max(attr_results.keys(), key=lambda k: attr_results[k]["attr_pos"].get("top20%", 0)) if attr_results else None
        if best_beta:
            r = attr_results[best_beta]
            L.log(f"    best β={best_beta}: attr_pos top20%={r['attr_pos'].get('top20%',0)}%, "
                  f"L0 top20%={r['L0_last'].get('top20%',0)}%, "
                  f"direct top20%={r['direct'].get('top20%',0)}%")
        
        results[attr] = attr_results
    
    return results

# ===================== P349: 多token属性的位置策略 =====================
def run_p349(model, tokenizer, device, model_name):
    """多token属性的位置策略"""
    L.log("=== P349: 多token属性的位置策略 ===")
    
    method = "embedding" if model_name == "qwen3" else "lm_head"
    beta_range = [1.0, 2.0, 3.0, 5.0, 8.0]
    
    # 选择一些可能被分词为多token的属性词
    test_attrs = STIMULI["color_attrs"] + STIMULI["taste_attrs"] + STIMULI["size_attrs"]
    
    results = {}
    
    for attr in test_attrs:
        # 检查属性词的token数量
        tok_ids = tokenizer.encode(attr, add_special_tokens=False)
        n_tokens = len(tok_ids)
        
        direction = get_attr_direction(model, tokenizer, attr, method)
        if direction is None:
            continue
        
        attr_results = {"n_tokens": n_tokens}
        test_nouns = ALL_NOUNS[:60]
        
        for strategy in ["first_token", "last_token", "all_tokens"]:
            for beta in beta_range:
                stats = {"top1": 0, "top3": 0, "top5": 0, "top10": 0, "top20": 0, "total": 0, "avg_rank": 0}
                
                for noun in test_nouns:
                    prompt = f"The {attr} {noun} is"
                    attr_positions = find_token_positions(tokenizer, prompt, attr)
                    
                    if len(attr_positions) == 0:
                        continue
                    
                    # 选择策略
                    if strategy == "first_token":
                        target_pos = [attr_positions[0]]
                    elif strategy == "last_token":
                        target_pos = [attr_positions[-1]]
                    else:  # all_tokens
                        target_pos = attr_positions
                    
                    try:
                        logits = intervene_at_position_and_get_logits(
                            model, tokenizer, device, prompt, direction, beta, target_pos)
                    except:
                        continue
                    
                    rank_info = check_rank_in_logits(logits, tokenizer, attr)
                    
                    if rank_info["hit"]:
                        stats["total"] += 1
                        stats["avg_rank"] += rank_info["rank"]
                        if rank_info["top1"]: stats["top1"] += 1
                        if rank_info["top3"]: stats["top3"] += 1
                        if rank_info["top5"]: stats["top5"] += 1
                        if rank_info["top10"]: stats["top10"] += 1
                        if rank_info["top20"]: stats["top20"] += 1
                
                if stats["total"] > 0:
                    key = f"{strategy}_b{beta}"
                    stats["top1%"] = round(stats["top1"] / stats["total"] * 100, 1)
                    stats["top3%"] = round(stats["top3"] / stats["total"] * 100, 1)
                    stats["top5%"] = round(stats["top5"] / stats["total"] * 100, 1)
                    stats["top10%"] = round(stats["top10"] / stats["total"] * 100, 1)
                    stats["top20%"] = round(stats["top20"] / stats["total"] * 100, 1)
                    stats["avg_rank"] = round(stats["avg_rank"] / stats["total"], 1)
                    attr_results[key] = stats
        
        # 报告
        if n_tokens > 1:
            L.log(f"  {attr}: {n_tokens} tokens")
        results[attr] = attr_results
    
    return results

# ===================== P350: 生成测试 =====================
def run_p350(model, tokenizer, device, model_name):
    """位置干预后的生成质量测试"""
    L.log("=== P350: 生成测试 - 位置干预后生成质量 ===")
    
    method = "embedding" if model_name == "qwen3" else "lm_head"
    gen_templates = ["The {word} is", "A {word} can be", "This {word} has"]
    
    # 选择部分属性测试
    test_attrs = STIMULI["color_attrs"][:6] + STIMULI["taste_attrs"][:6] + STIMULI["size_attrs"][:6]
    test_nouns = ALL_NOUNS[:20]
    
    results = {}
    
    for attr in test_attrs:
        direction = get_attr_direction(model, tokenizer, attr, method)
        if direction is None:
            continue
        
        attr_results = {}
        beta = 3.0  # 使用中等beta
        
        for position_type in ["attr_word", "last_token", "no_intervention"]:
            stats = {"attr_appeared": 0, "total": 0, "examples": []}
            
            for noun in test_nouns:
                for template in gen_templates:
                    if position_type == "attr_word":
                        prompt = template.format(word=f"{attr} {noun}")
                        target_positions = find_token_positions(tokenizer, prompt, attr)
                        if len(target_positions) == 0:
                            continue
                        gen_text = intervene_at_position_and_generate(
                            model, tokenizer, device, prompt, direction, beta, target_positions)
                    elif position_type == "last_token":
                        prompt = template.format(word=noun)
                        gen_text = intervene_at_position_and_generate(
                            model, tokenizer, device, prompt, direction, beta, None)
                    else:  # no_intervention
                        prompt = template.format(word=noun)
                        toks = tokenizer(prompt, return_tensors="pt").to(device)
                        with torch.no_grad():
                            gen_ids = model.generate(
                                toks.input_ids, max_new_tokens=25,
                                do_sample=False, temperature=1.0,
                                pad_token_id=tokenizer.eos_token_id)
                            gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
                    
                    stats["total"] += 1
                    if attr.lower() in gen_text.lower():
                        stats["attr_appeared"] += 1
                    
                    if len(stats["examples"]) < 3:
                        stats["examples"].append({
                            "prompt": prompt, "gen_text": gen_text[:200],
                            "position_type": position_type
                        })
            
            if stats["total"] > 0:
                stats["appearance_rate"] = round(stats["attr_appeared"] / stats["total"] * 100, 1)
            
            attr_results[position_type] = stats
        
        L.log(f"  {attr}: attr_word={attr_results.get('attr_word',{}).get('appearance_rate',0)}%, "
              f"last_token={attr_results.get('last_token',{}).get('appearance_rate',0)}%, "
              f"no_intervention={attr_results.get('no_intervention',{}).get('appearance_rate',0)}%")
        
        results[attr] = attr_results
    
    return results

# ===================== 主函数 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    model_name = args.model
    L.log(f"===== Phase LXIII: 位置感知语义干预 =====")
    L.log(f"模型: {model_name}")
    
    # 加载模型
    L.log("加载模型...")
    model, tokenizer, device = load_model(model_name)
    n_layers = len(model.model.layers) if hasattr(model, 'model') and hasattr(model.model, 'layers') else 0
    D = model.config.hidden_size
    
    L.log(f"模型: {model_name}, 层数: {n_layers}, 维度: {D}, 设备: {device}")
    
    # ========== P347: 不同位置干预对比 ==========
    L.log("P347: 不同位置干预对比...")
    p347_results = run_p347(model, tokenizer, device, model_name)
    
    # ========== P348: 属性词位置精确干预 ==========
    L.log("P348: 属性词位置精确干预...")
    p348_results = run_p348(model, tokenizer, device, model_name)
    
    # ========== P349: 多token属性的位置策略 ==========
    L.log("P349: 多token属性的位置策略...")
    p349_results = run_p349(model, tokenizer, device, model_name)
    
    # ========== P350: 生成测试 ==========
    L.log("P350: 生成测试...")
    p350_results = run_p350(model, tokenizer, device, model_name)
    
    # ========== 保存结果 ==========
    all_results = {
        "model": model_name,
        "n_layers": n_layers,
        "D": D,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "p347_position_comparison": p347_results,
        "p348_attr_position_intervention": p348_results,
        "p349_multi_token_strategy": p349_results,
        "p350_generation_test": p350_results,
    }
    
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = OUT_DIR / f"phase_lxiii_p347_350_{model_name}_{ts}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    L.log(f"结果已保存: {out_path}")
    
    # ========== 汇总 ==========
    L.log("\n===== 汇总 =====")
    
    # P347汇总
    L.log("\nP347: 不同位置干预对比")
    for attr in ALL_ATTRS[:12]:  # 只报告前12个
        if attr in p347_results:
            attr_res = p347_results[attr]
            # 找最佳位置
            best_pos = None
            best_top20 = 0
            for key, stats in attr_res.items():
                top20 = stats.get("top20%", 0)
                if top20 > best_top20:
                    best_top20 = top20
                    best_pos = key
            if best_pos:
                L.log(f"  {attr}: best={best_pos}, top20%={best_top20}%")
    
    # P348汇总
    L.log("\nP348: 属性词位置 vs L0 vs 直接输入")
    for attr in ALL_ATTRS[:12]:
        if attr in p348_results:
            attr_res = p348_results[attr]
            best_beta = max(attr_res.keys(), key=lambda k: attr_res[k]["attr_pos"].get("top20%", 0)) if attr_res else None
            if best_beta:
                r = attr_res[best_beta]
                L.log(f"  {attr}(β={best_beta}): pos={r['attr_pos'].get('top20%',0)}%, "
                      f"L0={r['L0_last'].get('top20%',0)}%, "
                      f"direct={r['direct'].get('top20%',0)}%")
    
    # P350汇总
    L.log("\nP350: 生成测试")
    for attr in list(p350_results.keys())[:12]:
        r = p350_results[attr]
        L.log(f"  {attr}: attr_pos={r.get('attr_word',{}).get('appearance_rate',0)}%, "
              f"last_tok={r.get('last_token',{}).get('appearance_rate',0)}%, "
              f"no_int={r.get('no_intervention',{}).get('appearance_rate',0)}%")
    
    # 释放GPU
    del model
    gc.collect()
    torch.cuda.empty_cache()
    L.log("GPU已释放")
    
    L.close()

if __name__ == "__main__":
    main()
