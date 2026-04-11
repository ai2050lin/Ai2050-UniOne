"""
Phase LX-P335/336/337/338: 颜色词信号方向修正
======================================================================

Phase LIX核心发现:
  1. pink/gray在GLM4 L34=100%! 颜色首次可操控!
  2. red在GLM4 L8=20%! 但大部分颜色词仍0%
  3. 颜色信号方向错误: 推到乱码token(のも/oe/ít)
  4. 基线排名高→难干预: 颜色词基线排名<7000
  5. huge/big/sour被全层扫描解锁

Phase LX核心目标: 为什么颜色信号指向错误方向? 如何修正?

四大实验:
  P335: Embedding方向干预
    - 用属性词的embedding作为信号方向
    - direction = embed(attr_word) - embed(参考词)
    - 假设: embedding空间比CCA信号更直接
    - 12属性 × 3参考词策略 × 5β = 180组合

  P336: lm_head权重方向干预
    - 直接从lm_head权重矩阵提取属性词的方向
    - direction = lm_head.weight[attr_token_id]
    - 这是最直接的"让模型输出某词"的方向
    - 12属性 × 5β = 60组合

  P337: 对比方向干预
    - 用"attr noun" vs "noun"的隐藏状态差异作为方向
    - direction = mean(h("red apple")) - mean(h("apple"))
    - 这是属性在隐藏空间中的实际编码方向
    - 12属性 × 全层 × 5β = 大规模

  P338: 最优方向组合
    - 用P335-337中最有效的方法
    - 全属性测试 + 最优层+β
    - 目标: red/green/blue > 20%

数据规模: 12属性 × 60名词 × 全层 × 多β × 3方法 × 3模型
实验模型: qwen3 -> glm4 -> deepseek7b (串行, 避免OOM)
"""

import torch
import numpy as np
import os, sys, gc, time, json, argparse
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
import warnings
warnings.filterwarnings('ignore')

import functools, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
print = functools.partial(print, flush=True)

OUT_DIR = Path("tests/glm5_temp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUT_DIR / "phase_lx_log.txt"

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

COLOR_LABELS = [(n, "color", c, NOUN_TO_FAMILY.get(n,"unknown")) for n in ALL_NOUNS for c in STIMULI["color_attrs"]]
TASTE_LABELS = [(n, "taste", t, NOUN_TO_FAMILY.get(n,"unknown")) for n in ALL_NOUNS for t in STIMULI["taste_attrs"]]
SIZE_LABELS = [(n, "size", s, NOUN_TO_FAMILY.get(n,"unknown")) for n in ALL_NOUNS for s in STIMULI["size_attrs"]]

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

def check_intervention_success(logits_new, logits_diff, tokenizer, target_attr, top_k=20):
    attr_tok_ids = tokenizer.encode(target_attr, add_special_tokens=False)
    if len(attr_tok_ids) == 0:
        return False, -1, 0.0
    attr_tok_id = attr_tok_ids[0]
    top_k_ids = np.argsort(logits_diff)[-top_k:][::-1]
    hit = attr_tok_id in top_k_ids
    rank = int(np.where(top_k_ids == attr_tok_id)[0][0]) + 1 if hit else -1
    target_logit_change = float(logits_diff[attr_tok_id]) if attr_tok_id < len(logits_diff) else 0.0
    return hit, rank, target_logit_change

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

# ===================== P335: Embedding方向干预 =====================
def run_p335(model, tokenizer, device, n_layers, D, model_name):
    """用属性词的embedding作为信号方向"""
    L.log("=== P335: Embedding方向干预 ===")
    
    beta_range = [1.0, 3.0, 5.0, 10.0, 20.0, 50.0]
    alpha_fixed = 0.5
    
    # 获取embedding层(不加载全部权重, 逐属性提取)
    embed_layer = model.get_input_embeddings()
    
    results = {}
    
    for attr_type in ["color", "size", "taste"]:
        attr_list = STIMULI[f"{attr_type}_attrs"]
        
        # Phase LIX发现的最优层
        best_layers_map = {
            "qwen3": {"color": 33, "taste": 9, "size": 9},
            "glm4": {"color": 34, "taste": 14, "size": 24},
            "deepseek7b": {"color": 9, "taste": 9, "size": 8},
        }
        layer = best_layers_map.get(model_name, {}).get(attr_type, n_layers // 2)
        
        # 收集名词隐藏状态
        noun_h = collect_noun_hidden_states(model, tokenizer, device, ALL_NOUNS[:20], PROMPT_TEMPLATES_30, layer)
        
        # 计算f_base
        all_noun_mean = np.mean(list(noun_h.values()), axis=0)
        noun_fbase = {n: m - all_noun_mean for n, m in noun_h.items()}
        
        attr_results = {}
        
        for attr in attr_list:
            attr_tok_ids = tokenizer.encode(attr, add_special_tokens=False)
            if len(attr_tok_ids) == 0:
                continue
            attr_tok_id = attr_tok_ids[0]
            
            # 方法1: 直接用属性词的embedding(逐个提取)
            direction = embed_layer.weight[attr_tok_id:attr_tok_id+1].detach().float().cpu().numpy()[0].copy()
            
            # 归一化
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm * 10.0  # 缩放到合理范围
            
            best_beta = 1.0
            best_rate = 0.0
            
            for beta in beta_range:
                total, success = 0, 0
                for noun, h_noun in noun_h.items():
                    f_base = noun_fbase.get(noun, np.zeros(D))
                    
                    logits_new, logits_diff = intervene_with_direction(
                        model, device, h_noun, direction, layer, 
                        f_base=f_base, alpha=alpha_fixed, beta=beta)
                    
                    hit, rank, logit_change = check_intervention_success(
                        logits_new, logits_diff, tokenizer, attr)
                    
                    if hit:
                        success += 1
                    total += 1
                
                rate = success / total * 100 if total > 0 else 0
                if rate > best_rate:
                    best_rate = rate
                    best_beta = beta
            
            attr_results[attr] = {"best_beta": best_beta, "best_rate": best_rate, "method": "embedding_direct"}
            if best_rate > 0:
                L.log(f"  {attr}(embedding): β={best_beta}, rate={best_rate:.1f}%")
        
        results[attr_type] = attr_results
    
    return results

# ===================== P336: lm_head权重方向干预 =====================
def run_p336(model, tokenizer, device, n_layers, D, model_name):
    """用lm_head权重矩阵的行作为信号方向"""
    L.log("=== P336: lm_head权重方向干预 ===")
    
    beta_range = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]
    alpha_fixed = 0.5
    
    # 获取lm_head(不加载全部权重, 逐属性提取)
    lm_head = model.lm_head if hasattr(model, 'lm_head') else model.get_output_embeddings()
    
    results = {}
    
    for attr_type in ["color", "size", "taste"]:
        attr_list = STIMULI[f"{attr_type}_attrs"]
        
        best_layers_map = {
            "qwen3": {"color": 33, "taste": 9, "size": 9},
            "glm4": {"color": 34, "taste": 14, "size": 24},
            "deepseek7b": {"color": 9, "taste": 9, "size": 8},
        }
        layer = best_layers_map.get(model_name, {}).get(attr_type, n_layers // 2)
        
        noun_h = collect_noun_hidden_states(model, tokenizer, device, ALL_NOUNS[:20], PROMPT_TEMPLATES_30, layer)
        all_noun_mean = np.mean(list(noun_h.values()), axis=0)
        noun_fbase = {n: m - all_noun_mean for n, m in noun_h.items()}
        
        attr_results = {}
        
        for attr in attr_list:
            attr_tok_ids = tokenizer.encode(attr, add_special_tokens=False)
            if len(attr_tok_ids) == 0:
                continue
            attr_tok_id = attr_tok_ids[0]
            
            # lm_head的行向量就是"从隐藏状态到某词logit"的投影方向
            direction = lm_head.weight[attr_tok_id:attr_tok_id+1].detach().float().cpu().numpy()[0].copy()
            
            # 归一化
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm * 10.0
            
            best_beta = 1.0
            best_rate = 0.0
            
            for beta in beta_range:
                total, success = 0, 0
                for noun, h_noun in noun_h.items():
                    f_base = noun_fbase.get(noun, np.zeros(D))
                    
                    logits_new, logits_diff = intervene_with_direction(
                        model, device, h_noun, direction, layer, 
                        f_base=f_base, alpha=alpha_fixed, beta=beta)
                    
                    hit, rank, logit_change = check_intervention_success(
                        logits_new, logits_diff, tokenizer, attr)
                    
                    if hit:
                        success += 1
                    total += 1
                
                rate = success / total * 100 if total > 0 else 0
                if rate > best_rate:
                    best_rate = rate
                    best_beta = beta
            
            attr_results[attr] = {"best_beta": best_beta, "best_rate": best_rate, "method": "lm_head_weight"}
            if best_rate > 0:
                L.log(f"  {attr}(lm_head): β={best_beta}, rate={best_rate:.1f}%")
        
        results[attr_type] = attr_results
    
    return results

# ===================== P337: 对比方向干预 =====================
def run_p337(model, tokenizer, device, n_layers, D, model_name):
    """用'attr noun' vs 'noun'的隐藏状态差异作为方向"""
    L.log("=== P337: 对比方向干预 ===")
    
    beta_range = [1.0, 3.0, 5.0, 10.0, 20.0]
    alpha_fixed = 0.5
    layer_step = 2
    
    results = {}
    
    for attr_type in ["color", "size", "taste"]:
        attr_list = STIMULI[f"{attr_type}_attrs"]
        test_nouns = STIMULI["fruit_family"][:5]  # 5个名词计算对比方向
        
        # 在每层计算对比方向
        best_per_attr = {}
        
        for layer in range(0, n_layers, layer_step):
            # 收集"attr noun"和"noun"的隐藏状态
            contrast_dirs = {}
            
            for attr in attr_list:
                h_attr_noun_list = []
                h_noun_list = []
                
                for noun in test_nouns:
                    # "attr noun"
                    text_attr = f"The {attr} {noun} is"
                    toks = tokenizer(text_attr, return_tensors="pt").to(device)
                    with torch.no_grad():
                        out = model(toks.input_ids, output_hidden_states=True)
                        h_attr_noun_list.append(out.hidden_states[layer][0, -1].float().cpu().numpy())
                    
                    # "noun"
                    text_noun = f"The {noun} is"
                    toks = tokenizer(text_noun, return_tensors="pt").to(device)
                    with torch.no_grad():
                        out = model(toks.input_ids, output_hidden_states=True)
                        h_noun_list.append(out.hidden_states[layer][0, -1].float().cpu().numpy())
                
                # 对比方向 = mean(h("attr noun")) - mean(h("noun"))
                direction = np.mean(h_attr_noun_list, axis=0) - np.mean(h_noun_list, axis=0)
                contrast_dirs[attr] = direction
            
            # 在该层测试
            noun_h = collect_noun_hidden_states(model, tokenizer, device, ALL_NOUNS[:10], PROMPT_TEMPLATES_30, layer)
            all_noun_mean = np.mean(list(noun_h.values()), axis=0)
            noun_fbase = {n: m - all_noun_mean for n, m in noun_h.items()}
            
            for attr in attr_list:
                direction = contrast_dirs[attr]
                norm = np.linalg.norm(direction)
                if norm < 1e-6:
                    continue
                direction = direction / norm * 10.0
                
                for beta in beta_range:
                    total, success = 0, 0
                    for noun, h_noun in noun_h.items():
                        f_base = noun_fbase.get(noun, np.zeros(D))
                        
                        logits_new, logits_diff = intervene_with_direction(
                            model, device, h_noun, direction, layer, 
                            f_base=f_base, alpha=alpha_fixed, beta=beta)
                        
                        hit, rank, logit_change = check_intervention_success(
                            logits_new, logits_diff, tokenizer, attr)
                        
                        if hit:
                            success += 1
                        total += 1
                    
                    rate = success / total * 100 if total > 0 else 0
                    
                    if attr not in best_per_attr or rate > best_per_attr[attr]["best_rate"]:
                        best_per_attr[attr] = {
                            "best_layer": layer, "best_beta": beta,
                            "best_rate": rate, "method": "contrast"
                        }
                    
                    if rate > 0 and rate >= best_per_attr[attr]["best_rate"]:
                        L.log(f"  {attr} L{layer} β={beta}(contrast): {rate:.1f}%")
        
        for attr, data in best_per_attr.items():
            L.log(f"  {attr}(contrast最优): L{data['best_layer']}, β={data['best_beta']}, rate={data['best_rate']:.1f}%")
        
        results[attr_type] = best_per_attr
    
    return results

# ===================== P338: 最优方向组合测试 =====================
def run_p338(model, tokenizer, device, n_layers, D, model_name, p335_res, p336_res, p337_res):
    """用P335-337中最有效的方法做完整测试"""
    L.log("=== P338: 最优方向组合测试 ===")
    
    beta_range = [1.0, 3.0, 5.0, 10.0, 20.0]
    alpha_fixed = 0.5
    
    # 获取embedding和lm_head权重(逐属性加载, 避免OOM)
    embed_layer = model.get_input_embeddings()
    lm_head = model.lm_head if hasattr(model, 'lm_head') else model.get_output_embeddings()
    
    results = {}
    
    for attr_type in ["color", "size", "taste"]:
        attr_list = STIMULI[f"{attr_type}_attrs"]
        
        best_layers_map = {
            "qwen3": {"color": 33, "taste": 9, "size": 9},
            "glm4": {"color": 34, "taste": 14, "size": 24},
            "deepseek7b": {"color": 9, "taste": 9, "size": 8},
        }
        layer = best_layers_map.get(model_name, {}).get(attr_type, n_layers // 2)
        
        noun_h = collect_noun_hidden_states(model, tokenizer, device, ALL_NOUNS, PROMPT_TEMPLATES_30, layer)
        all_noun_mean = np.mean(list(noun_h.values()), axis=0)
        noun_fbase = {n: m - all_noun_mean for n, m in noun_h.items()}
        
        attr_best = {}
        
        for attr in attr_list:
            attr_tok_ids = tokenizer.encode(attr, add_special_tokens=False)
            if len(attr_tok_ids) == 0:
                continue
            attr_tok_id = attr_tok_ids[0]
            
            # 3种方向
            directions = {}
            
            # 1. Embedding方向(逐个提取, 避免OOM)
            d1 = embed_layer.weight[attr_tok_id:attr_tok_id+1].detach().float().cpu().numpy()[0].copy()
            n1 = np.linalg.norm(d1)
            if n1 > 0:
                directions["embedding"] = d1 / n1 * 10.0
            
            # 2. lm_head方向(逐个提取, 避免OOM)
            d2 = lm_head.weight[attr_tok_id:attr_tok_id+1].detach().float().cpu().numpy()[0].copy()
            n2 = np.linalg.norm(d2)
            if n2 > 0:
                directions["lm_head"] = d2 / n2 * 10.0
            
            # 3. 对比方向(在该层计算)
            h_attr_list = []
            h_noun_list = []
            for noun in STIMULI["fruit_family"][:5]:
                text_a = f"The {attr} {noun} is"
                toks = tokenizer(text_a, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model(toks.input_ids, output_hidden_states=True)
                    h_attr_list.append(out.hidden_states[layer][0, -1].float().cpu().numpy())
                
                text_n = f"The {noun} is"
                toks = tokenizer(text_n, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model(toks.input_ids, output_hidden_states=True)
                    h_noun_list.append(out.hidden_states[layer][0, -1].float().cpu().numpy())
            
            d3 = np.mean(h_attr_list, axis=0) - np.mean(h_noun_list, axis=0)
            n3 = np.linalg.norm(d3)
            if n3 > 1e-6:
                directions["contrast"] = d3 / n3 * 10.0
            
            # 每种方向扫描β
            best_method = "none"
            best_beta = 1.0
            best_rate = 0.0
            
            for method_name, direction in directions.items():
                for beta in beta_range:
                    total, success = 0, 0
                    for noun, h_noun in noun_h.items():
                        f_base = noun_fbase.get(noun, np.zeros(D))
                        
                        logits_new, logits_diff = intervene_with_direction(
                            model, device, h_noun, direction, layer, 
                            f_base=f_base, alpha=alpha_fixed, beta=beta)
                        
                        hit, rank, logit_change = check_intervention_success(
                            logits_new, logits_diff, tokenizer, attr)
                        
                        if hit:
                            success += 1
                        total += 1
                    
                    rate = success / total * 100 if total > 0 else 0
                    if rate > best_rate:
                        best_rate = rate
                        best_beta = beta
                        best_method = method_name
            
            attr_best[attr] = {
                "best_method": best_method, "best_beta": best_beta, 
                "best_rate": best_rate, "total": total
            }
            L.log(f"  {attr}: 最优方法={best_method}, β={best_beta}, rate={best_rate:.1f}%")
        
        results[attr_type] = attr_best
    
    return results

# ===================== 主函数 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    model_name = args.model
    L.log(f"=== Phase LX: 颜色词信号方向修正 — {model_name} ===")
    
    L.log(f"加载模型: {model_name}")
    model, tokenizer, device = load_model(model_name)
    n_layers = model.config.num_hidden_layers + 1
    D = model.config.hidden_size
    L.log(f"模型: {model_name}, 层数={n_layers}, 维度={D}")
    
    # ========== P335: Embedding方向 ==========
    L.log("P335: Embedding方向干预...")
    p335_results = run_p335(model, tokenizer, device, n_layers, D, model_name)
    
    # ========== P336: lm_head权重方向 ==========
    L.log("P336: lm_head权重方向干预...")
    p336_results = run_p336(model, tokenizer, device, n_layers, D, model_name)
    
    # ========== P337: 对比方向 ==========
    L.log("P337: 对比方向干预...")
    p337_results = run_p337(model, tokenizer, device, n_layers, D, model_name)
    
    # ========== P338: 最优方向组合 ==========
    L.log("P338: 最优方向组合测试...")
    p338_results = run_p338(model, tokenizer, device, n_layers, D, model_name, p335_results, p336_results, p337_results)
    
    # ========== 保存结果 ==========
    all_results = {
        "model": model_name,
        "n_layers": n_layers,
        "D": D,
        "p335_embedding_dir": p335_results,
        "p336_lm_head_dir": p336_results,
        "p337_contrast_dir": p337_results,
        "p338_best_combo": p338_results,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M"),
    }
    
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = OUT_DIR / f"phase_lx_p335_338_{model_name}_{ts}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    L.log(f"结果已保存: {out_path}")
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    L.log("GPU已清理")

if __name__ == "__main__":
    main()
