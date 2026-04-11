"""
Phase LIX-P331/332/333/334: 属性不可干预根因分析
======================================================================

Phase LVIII核心突破:
  1. 反义词信号算子几乎正交(余弦≈0): DNN语义是"坐标式"非"光谱式"
  2. 逐属性β优化后: GLM4 10个100%, Qwen3 7个, DS7B 6个
  3. A级属性: short/heavy/fresh(3模型100%)
  4. D级属性: red/green/blue/black/pink/purple/brown/gray(全0%)

Phase LIX核心目标: 为什么C/D级属性0%? 是层不对还是β不够?

四大实验:
  P331: 全层逐属性扫描
    - 每个属性×每2层×5个β = 大规模扫描
    - 12属性 × ~18层 × 5β = ~1080测试组合
    - 目标: 找到C/D级属性的最优层+β

  P332: C/D级属性深度扫描
    - 对big/tall/huge/red/blue等0%属性
    - β扫描到100, 层扫描所有层
    - 目标: 极端参数下是否可达>0%

  P333: 属性词基线logit分析
    - 不干预时, 属性词在top-k中的排名
    - 分析0%属性是否因为基线排名太低
    - 假设: 如果基线top-1就是属性词, 干预空间很小

  P334: 干预效果验证
    - 对100%属性(short/heavy/fresh)的干预
    - 检查干预后的top-5词变化
    - 验证干预是否真的改变了语义

数据规模: 12属性×18层×5β×60名词 × 3模型
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
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

import functools, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
print = functools.partial(print, flush=True)

OUT_DIR = Path("tests/glm5_temp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUT_DIR / "phase_lix_log.txt"

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

# ===================== 数据集定义 =====================
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

# Phase LVIII发现的属性分级
ATTR_LEVELS = {
    "size": {
        "A": ["short", "heavy", "long", "small", "thin"],  # ≥2模型100%
        "B": ["wide", "tiny", "light"],                     # 1模型100%
        "C": ["big", "tall", "huge", "thick"],              # 无模型100%
    },
    "taste": {
        "A": ["fresh", "soft", "sweet"],
        "B": ["rich"],
        "C": ["sour", "bitter", "salty", "spicy", "crisp", "tart", "savory", "mild"],
    },
    "color": {
        "A": ["orange", "white", "gold"],
        "B": [],
        "C": ["red", "green", "yellow", "brown", "blue", "black", "pink", "purple", "gray"],
    },
}

def noun_centered_G(G, labels):
    nouns = [l[0] for l in labels]
    unique_nouns = sorted(set(nouns))
    noun_means = {}
    for n in unique_nouns:
        mask = [i for i, x in enumerate(nouns) if x == n]
        noun_means[n] = np.mean(G[mask], axis=0)
    G_centered = np.array([G[i] - noun_means[nouns[i]] for i in range(len(G))])
    return G_centered, noun_means

def build_indicator(labels, attr_list, attr_type):
    N = len(labels)
    K = len(attr_list)
    Y = np.zeros((N, K))
    attr_to_idx = {a: i for i, a in enumerate(attr_list)}
    for i, l in enumerate(labels):
        if l[1] == attr_type:
            Y[i, attr_to_idx[l[2]]] = 1.0
    return Y

def compute_signal_subspace(G_centered, Y, n_cca=10):
    N, D = G_centered.shape
    K = Y.shape[1]
    pca_dim = min(30, N - 1, D)
    pca = PCA(n_components=pca_dim)
    G_pca = pca.fit_transform(G_centered)
    if np.any(np.isnan(G_pca)):
        return None, None, None, None
    n_cca = min(n_cca, K - 1, G_pca.shape[1] - 1, N - 1)
    if n_cca < 2:
        return None, None, None, None
    try:
        cca = CCA(n_components=n_cca, max_iter=500)
        cca.fit(G_pca, Y)
    except:
        return None, None, None, None
    U = cca.x_weights_
    P_s_pca = U @ np.linalg.inv(U.T @ U + 1e-6 * np.eye(n_cca)) @ U.T
    return pca, cca, P_s_pca, U

def extract_signal_operator(G_centered, labels, attr, attr_type, pca, P_s_pca):
    attr_mask = [i for i, l in enumerate(labels) if l[2] == attr]
    if len(attr_mask) < 2:
        return None
    G_attr = G_centered[attr_mask]
    G_attr_mean = np.mean(G_attr, axis=0)
    G_pca = pca.transform(G_attr_mean.reshape(1, -1))
    G_signal_pca = P_s_pca @ G_pca.T
    G_signal = pca.inverse_transform(G_signal_pca.T)[0]
    return G_signal

def intervene_with_operator(model, device, h_noun, operator, layer, f_base=None, alpha=1.0, beta=1.0):
    h_new = h_noun.copy()
    if f_base is not None:
        h_new = h_new + alpha * f_base
    h_new = h_new + beta * operator
    
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

# ===================== 采集隐藏状态 =====================
def collect_hidden_states(model, tokenizer, device, labels, templates):
    N = len(labels)
    n_layers = model.config.num_hidden_layers + 1
    sample_text = templates[0].format(word=labels[0][0])
    toks = tokenizer(sample_text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(toks.input_ids, output_hidden_states=True)
    D = out.hidden_states[0].shape[-1]
    
    all_H = [np.zeros((N, D), dtype=np.float32) for _ in range(n_layers)]
    for i, (noun, attr_type, attr, family) in enumerate(labels):
        word = f"{attr} {noun}"
        template = templates[i % len(templates)]
        text = template.format(word=word)
        toks = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(toks.input_ids, output_hidden_states=True)
        for l in range(n_layers):
            all_H[l][i] = out.hidden_states[l][0, -1].float().cpu().numpy()
        if (i + 1) % 200 == 0:
            L.log(f"  采集进度: {i+1}/{N}")
    return all_H, n_layers, D

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

# ===================== P331: 全层逐属性扫描 =====================
def run_p331(model, tokenizer, device, all_H_dict, n_layers, D, labels_dict, model_name):
    """全层逐属性扫描: 每个属性×每2层×5β"""
    L.log("=== P331: 全层逐属性扫描 ===")
    
    beta_range = [1.0, 3.0, 5.0, 10.0, 20.0]
    alpha_fixed = 0.5
    layer_step = 2
    
    results = {}
    
    for attr_type in ["size", "taste", "color"]:
        labels = labels_dict[attr_type]
        attr_list = STIMULI[f"{attr_type}_attrs"]
        
        # 每层计算信号子空间和算子
        layer_attr_results = {}
        
        for layer in range(0, n_layers, layer_step):
            H_layer = all_H_dict[attr_type][layer]
            G = H_layer
            G_centered, noun_means = noun_centered_G(G, labels)
            Y = build_indicator(labels, attr_list, attr_type)
            
            pca, cca, P_s, U = compute_signal_subspace(G_centered, Y, n_cca=min(10, len(attr_list)-1))
            if pca is None:
                continue
            
            global_signal_ops = {}
            for attr in attr_list:
                op = extract_signal_operator(G_centered, labels, attr, attr_type, pca, P_s)
                if op is not None:
                    global_signal_ops[attr] = op
            
            all_noun_mean = np.mean(list(noun_means.values()), axis=0)
            noun_fbase = {n: m - all_noun_mean for n, m in noun_means.items()}
            
            # 每个属性×每个β
            attr_layer_beta = {}
            for test_attr in attr_list:
                if test_attr not in global_signal_ops:
                    continue
                
                best_beta = 1.0
                best_rate = 0.0
                
                for beta in beta_range:
                    total, success = 0, 0
                    # 每族2个名词 = 10测试(加速)
                    for fam in FAMILY_NAMES:
                        for test_noun in STIMULI[fam][:2]:
                            h_noun_data = noun_means.get(test_noun)
                            if h_noun_data is None:
                                continue
                            f_base_noun = noun_fbase.get(test_noun, np.zeros(D))
                            
                            logits_new, logits_diff = intervene_with_operator(
                                model, device, h_noun_data, global_signal_ops[test_attr],
                                layer, f_base=f_base_noun, alpha=alpha_fixed, beta=beta)
                            
                            hit, rank, logit_change = check_intervention_success(
                                logits_new, logits_diff, tokenizer, test_attr)
                            
                            if hit:
                                success += 1
                            total += 1
                    
                    rate = success / total * 100 if total > 0 else 0
                    if rate > best_rate:
                        best_rate = rate
                        best_beta = beta
                
                attr_layer_beta[test_attr] = {"best_beta": best_beta, "best_rate": best_rate}
            
            layer_attr_results[layer] = attr_layer_beta
            
            # 只报告C/D级属性的最佳层
            for level_name, level_attrs in ATTR_LEVELS.get(attr_type, {}).items():
                if level_name in ["C"]:
                    for attr in level_attrs:
                        if attr in attr_layer_beta and attr_layer_beta[attr]["best_rate"] > 0:
                            L.log(f"  L{layer} {attr}(C级): β={attr_layer_beta[attr]['best_beta']}, rate={attr_layer_beta[attr]['best_rate']:.1f}%")
        
        # 汇总: 每个属性的最优层+β
        attr_best = {}
        for attr in attr_list:
            best_layer = -1
            best_beta = 1.0
            best_rate = 0.0
            for layer, layer_data in layer_attr_results.items():
                if attr in layer_data and layer_data[attr]["best_rate"] > best_rate:
                    best_rate = layer_data[attr]["best_rate"]
                    best_beta = layer_data[attr]["best_beta"]
                    best_layer = layer
            attr_best[attr] = {"best_layer": best_layer, "best_beta": best_beta, "best_rate": best_rate}
            level = "A"
            for ln, la in ATTR_LEVELS.get(attr_type, {}).items():
                if attr in la:
                    level = ln
            if best_rate > 0 or level == "C":
                L.log(f"  {attr}({level}级): 最优L{best_layer}, β={best_beta}, rate={best_rate:.1f}%")
        
        results[attr_type] = {"attr_best": attr_best, "layer_attr_results": layer_attr_results}
    
    return results

# ===================== P332: C/D级属性深度扫描 =====================
def run_p332(model, tokenizer, device, all_H_dict, n_layers, D, labels_dict, model_name):
    """C/D级属性深度扫描: β到100, 全层"""
    L.log("=== P332: C/D级属性深度扫描 ===")
    
    # 只测C/D级属性
    c_d_attrs = {
        "size": ["big", "tall", "huge", "thick"],
        "taste": ["sour", "bitter", "salty", "spicy"],
        "color": ["red", "green", "blue", "black", "pink", "purple", "brown", "gray"],
    }
    
    beta_range_deep = [1.0, 3.0, 5.0, 10.0, 20.0, 30.0, 50.0, 80.0, 100.0]
    alpha_fixed = 0.5
    layer_step = 2
    
    results = {}
    
    for attr_type, c_attrs in c_d_attrs.items():
        labels = labels_dict[attr_type]
        attr_list = STIMULI[f"{attr_type}_attrs"]
        
        attr_best = {}
        
        for layer in range(0, n_layers, layer_step):
            H_layer = all_H_dict[attr_type][layer]
            G = H_layer
            G_centered, noun_means = noun_centered_G(G, labels)
            Y = build_indicator(labels, attr_list, attr_type)
            
            pca, cca, P_s, U = compute_signal_subspace(G_centered, Y, n_cca=min(10, len(attr_list)-1))
            if pca is None:
                continue
            
            global_signal_ops = {}
            for attr in c_attrs:
                op = extract_signal_operator(G_centered, labels, attr, attr_type, pca, P_s)
                if op is not None:
                    global_signal_ops[attr] = op
            
            all_noun_mean = np.mean(list(noun_means.values()), axis=0)
            noun_fbase = {n: m - all_noun_mean for n, m in noun_means.items()}
            
            for test_attr in c_attrs:
                if test_attr not in global_signal_ops:
                    continue
                
                for beta in beta_range_deep:
                    total, success = 0, 0
                    for fam in FAMILY_NAMES:
                        for test_noun in STIMULI[fam][:2]:
                            h_noun_data = noun_means.get(test_noun)
                            if h_noun_data is None:
                                continue
                            f_base_noun = noun_fbase.get(test_noun, np.zeros(D))
                            
                            logits_new, logits_diff = intervene_with_operator(
                                model, device, h_noun_data, global_signal_ops[test_attr],
                                layer, f_base=f_base_noun, alpha=alpha_fixed, beta=beta)
                            
                            hit, rank, logit_change = check_intervention_success(
                                logits_new, logits_diff, tokenizer, test_attr)
                            
                            if hit:
                                success += 1
                            total += 1
                    
                    rate = success / total * 100 if total > 0 else 0
                    
                    if test_attr not in attr_best or rate > attr_best[test_attr]["best_rate"]:
                        attr_best[test_attr] = {
                            "best_layer": layer, "best_beta": beta, 
                            "best_rate": rate, "total": total, "success": success
                        }
                    
                    if rate > 0:
                        L.log(f"  {test_attr} L{layer} β={beta}: {rate:.1f}%")
        
        for attr, data in attr_best.items():
            L.log(f"  {attr} 最优: L{data['best_layer']}, β={data['best_beta']}, rate={data['best_rate']:.1f}%")
        
        results[attr_type] = attr_best
    
    return results

# ===================== P333: 属性词基线logit分析 =====================
def run_p333(model, tokenizer, device, n_layers):
    """属性词基线logit分析: 不干预时属性词的排名"""
    L.log("=== P333: 属性词基线logit分析 ===")
    
    results = {}
    
    for attr_type in ["size", "taste", "color"]:
        attr_list = STIMULI[f"{attr_type}_attrs"]
        test_nouns = STIMULI["fruit_family"][:3]  # 只测3个名词
        
        attr_baseline = {}
        
        for attr in attr_list:
            ranks = []
            logit_values = []
            
            for noun in test_nouns:
                text = f"The {noun} is"
                toks = tokenizer(text, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model(toks.input_ids, output_hidden_states=True)
                    logits = out.logits[0, -1].float().cpu().numpy()
                
                # 属性词的logit
                attr_tok_ids = tokenizer.encode(attr, add_special_tokens=False)
                if len(attr_tok_ids) == 0:
                    continue
                attr_tok_id = attr_tok_ids[0]
                
                attr_logit = float(logits[attr_tok_id]) if attr_tok_id < len(logits) else -100
                # 排名
                sorted_ids = np.argsort(logits)[::-1]
                rank = int(np.where(sorted_ids == attr_tok_id)[0][0]) + 1 if attr_tok_id in sorted_ids else -1
                
                ranks.append(rank)
                logit_values.append(attr_logit)
            
            if ranks:
                avg_rank = np.mean(ranks)
                avg_logit = np.mean(logit_values)
                attr_baseline[attr] = {
                    "avg_rank": float(avg_rank),
                    "avg_logit": float(avg_logit),
                }
                L.log(f"  {attr}: 基线排名={avg_rank:.0f}, logit={avg_logit:.2f}")
        
        results[attr_type] = attr_baseline
    
    return results

# ===================== P334: 干预效果验证 =====================
def run_p334(model, tokenizer, device, all_H_dict, n_layers, D, labels_dict, best_layers):
    """干预效果验证: 检查top-5词变化"""
    L.log("=== P334: 干预效果验证 ===")
    
    # 测试A级属性
    test_attrs = ["short", "heavy", "fresh", "big", "red"]
    alpha_fixed = 0.5
    beta_map = {"short": 2.0, "heavy": 2.0, "fresh": 3.0, "big": 10.0, "red": 10.0}
    
    results = {}
    
    for attr_type in ["size", "taste", "color"]:
        labels = labels_dict[attr_type]
        attr_list = STIMULI[f"{attr_type}_attrs"]
        layer = best_layers.get(attr_type, n_layers // 2)
        
        # 过滤测试属性
        test_attrs_filtered = [a for a in test_attrs if a in attr_list]
        if not test_attrs_filtered:
            continue
        
        H_layer = all_H_dict[attr_type][layer]
        G = H_layer
        G_centered, noun_means = noun_centered_G(G, labels)
        Y = build_indicator(labels, attr_list, attr_type)
        
        pca, cca, P_s, U = compute_signal_subspace(G_centered, Y, n_cca=min(10, len(attr_list)-1))
        if pca is None:
            continue
        
        global_signal_ops = {}
        for attr in test_attrs_filtered:
            op = extract_signal_operator(G_centered, labels, attr, attr_type, pca, P_s)
            if op is not None:
                global_signal_ops[attr] = op
        
        all_noun_mean = np.mean(list(noun_means.values()), axis=0)
        noun_fbase = {n: m - all_noun_mean for n, m in noun_means.items()}
        
        for test_attr in test_attrs_filtered:
            if test_attr not in global_signal_ops:
                continue
            
            beta = beta_map.get(test_attr, 5.0)
            
            # 选3个名词做详细分析
            for test_noun in ["apple", "cat", "car"][:1]:
                h_noun_data = noun_means.get(test_noun)
                if h_noun_data is None:
                    continue
                f_base_noun = noun_fbase.get(test_noun, np.zeros(D))
                
                # 基线
                with torch.no_grad():
                    h_base = torch.tensor(h_noun_data, dtype=torch.bfloat16).unsqueeze(0).unsqueeze(0).to(device)
                    lm_head = model.lm_head if hasattr(model, 'lm_head') else model.get_output_embeddings()
                    norm = model.model.norm if hasattr(model, 'model') and hasattr(model.model, 'norm') else None
                    if norm is not None:
                        logits_base = lm_head(norm(h_base))[0, 0].float().cpu().numpy()
                    else:
                        logits_base = lm_head(h_base)[0, 0].float().cpu().numpy()
                
                # 干预后
                logits_new, logits_diff = intervene_with_operator(
                    model, device, h_noun_data, global_signal_ops[test_attr],
                    layer, f_base=f_base_noun, alpha=alpha_fixed, beta=beta)
                
                # top-5基线词
                top5_base_ids = np.argsort(logits_base)[-5:][::-1]
                top5_base_words = [tokenizer.decode([i]) for i in top5_base_ids]
                top5_base_logits = [float(logits_base[i]) for i in top5_base_ids]
                
                # top-5干预词(按diff)
                top5_diff_ids = np.argsort(logits_diff)[-5:][::-1]
                top5_diff_words = [tokenizer.decode([i]) for i in top5_diff_ids]
                top5_diff_values = [float(logits_diff[i]) for i in top5_diff_ids]
                
                # 目标属性词的变化
                attr_tok_ids = tokenizer.encode(test_attr, add_special_tokens=False)
                attr_tok_id = attr_tok_ids[0] if attr_tok_ids else -1
                attr_logit_base = float(logits_base[attr_tok_id]) if attr_tok_id < len(logits_base) else -100
                attr_logit_new = float(logits_new[attr_tok_id]) if attr_tok_id < len(logits_new) else -100
                
                result_entry = {
                    "noun": test_noun,
                    "attr": test_attr,
                    "beta": beta,
                    "layer": layer,
                    "baseline_top5": list(zip(top5_base_words, top5_base_logits)),
                    "diff_top5": list(zip(top5_diff_words, top5_diff_values)),
                    "target_attr_logit_base": attr_logit_base,
                    "target_attr_logit_new": attr_logit_new,
                    "target_attr_delta": attr_logit_new - attr_logit_base,
                }
                
                L.log(f"  {test_attr}+{test_noun}(β={beta}): 基线logit={attr_logit_base:.2f}→{attr_logit_new:.2f}(Δ={attr_logit_new-attr_logit_base:.2f})")
                L.log(f"    diff_top5: {', '.join(f'{w}({v:.2f})' for w,v in result_entry['diff_top5'])}")
                
                if test_attr not in results:
                    results[test_attr] = []
                results[test_attr].append(result_entry)
    
    return results

# ===================== 主函数 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    model_name = args.model
    L.log(f"=== Phase LIX: 属性不可干预根因分析 — {model_name} ===")
    
    # 加载模型
    L.log(f"加载模型: {model_name}")
    model, tokenizer, device = load_model(model_name)
    n_layers = model.config.num_hidden_layers + 1
    D = model.config.hidden_size
    L.log(f"模型: {model_name}, 层数={n_layers}, 维度={D}")
    
    # 最优层
    best_layers_map = {
        "qwen3": {"color": 33, "taste": 9, "size": 9},
        "glm4": {"color": 37, "taste": 39, "size": 33},
        "deepseek7b": {"color": 9, "taste": 9, "size": 9},
    }
    best_layers = best_layers_map.get(model_name, {"color": n_layers//2, "taste": n_layers//2, "size": n_layers//2})
    
    # 采集隐藏状态
    labels_dict = {
        "color": COLOR_LABELS,
        "taste": TASTE_LABELS,
        "size": SIZE_LABELS,
    }
    
    L.log("采集属性+名词组合的隐藏状态...")
    all_H_color, _, _ = collect_hidden_states(model, tokenizer, device, COLOR_LABELS, PROMPT_TEMPLATES_30)
    all_H_taste, _, _ = collect_hidden_states(model, tokenizer, device, TASTE_LABELS, PROMPT_TEMPLATES_30)
    all_H_size, _, _ = collect_hidden_states(model, tokenizer, device, SIZE_LABELS, PROMPT_TEMPLATES_30)
    
    all_H_dict = {"color": all_H_color, "taste": all_H_taste, "size": all_H_size}
    
    # ========== P331: 全层逐属性扫描 ==========
    L.log("P331: 全层逐属性扫描...")
    p331_results = run_p331(model, tokenizer, device, all_H_dict, n_layers, D, labels_dict, model_name)
    
    # ========== P332: C/D级属性深度扫描 ==========
    L.log("P332: C/D级属性深度扫描...")
    p332_results = run_p332(model, tokenizer, device, all_H_dict, n_layers, D, labels_dict, model_name)
    
    # ========== P333: 属性词基线logit分析 ==========
    L.log("P333: 属性词基线logit分析...")
    p333_results = run_p333(model, tokenizer, device, n_layers)
    
    # ========== P334: 干预效果验证 ==========
    L.log("P334: 干预效果验证...")
    p334_results = run_p334(model, tokenizer, device, all_H_dict, n_layers, D, labels_dict, best_layers)
    
    # ========== 保存结果 ==========
    all_results = {
        "model": model_name,
        "n_layers": n_layers,
        "D": D,
        "p331_all_layer_attr": p331_results,
        "p332_cd_deep_scan": p332_results,
        "p333_baseline_logit": p333_results,
        "p334_intervention_verify": p334_results,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M"),
    }
    
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = OUT_DIR / f"phase_lix_p331_334_{model_name}_{ts}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    L.log(f"结果已保存: {out_path}")
    
    # 清理GPU
    del model
    gc.collect()
    torch.cuda.empty_cache()
    L.log("GPU已清理")

if __name__ == "__main__":
    main()
