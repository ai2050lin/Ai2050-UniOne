"""
Phase LVII-P322/323/324/325: 名词级f_base + 大规模β扫描
======================================================================

Phase LVI核心结论:
  1. norm层修复: Qwen3从0%→33.3%, DS7B→57.5%
  2. DS7B L9大小60.8%(α=3.0, β=0.5): 目前最高
  3. f_base必须名词级: DS7B P321全0%(用族平均)
  4. GLM4 β=5.0最优: 信号算子需要放大
  5. 热点层: Qwen3/DS7B L9, GLM4 L33

Phase LVII核心目标: 名词级f_base + 更大β扫描, 突破80%成功率

四大实验:
  P322: 名词级f_base跨族泛化 — 修复DS7B全0%问题
    - 用noun级别的f_base替代family_fbase
    - f_base = h(attr+noun) - h(noun) 在训练集上的均值
    - 5族×12名词×12属性×3类型 = 2160个干预

  P323: 大规模β扫描 — β∈{0.1,0.3,0.5,1,2,3,5,8,10,15,20}
    - 在最优层, 用名词级f_base, 扫描β和α
    - α∈{0.5,1,2,3,5}, β∈{0.1,0.3,0.5,1,2,3,5,8,10,15,20}
    - 目标: 找到最优α,β组合, 大小>80%

  P324: 全层+β联合扫描 — 每层找最优β
    - 每层扫描β∈{1,3,5,10,20}
    - 5族×4名词×6属性 = 120测试/层/β
    - 目标: 找到全局最优层+β组合

  P325: 12属性完整测试 — 验证干预覆盖面
    - 用全部12个属性(非前6个)
    - 5族×12名词×12属性 = 720测试/类型
    - 目标: 验证干预成功率在更多属性上是否稳定

数据规模: 2160三元组 × 30模板 + 名词级f_base + 大β网格
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
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import functools, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
print = functools.partial(print, flush=True)

OUT_DIR = Path("tests/glm5_temp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUT_DIR / "phase_lvii_log.txt"

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

def noun_centered_G(G, labels):
    """去名词均值: G_centered = G - mean(G|noun)"""
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
    """含norm层处理的干预"""
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
    """收集名词在指定层的隐藏状态(30模板平均)"""
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

# ===================== P322: 名词级f_base跨族泛化 =====================
def run_p322(model, tokenizer, device, all_H_dict, n_layers, D, labels_dict, best_layers, model_name):
    """名词级f_base跨族泛化: 修复DS7B全0%问题"""
    L.log("=== P322: 名词级f_base跨族泛化 ===")
    
    results = {}
    test_attrs = {
        "color": STIMULI["color_attrs"][:6],
        "taste": STIMULI["taste_attrs"][:6],
        "size": STIMULI["size_attrs"][:6],
    }
    
    for attr_type in ["color", "taste", "size"]:
        labels = labels_dict[attr_type]
        attr_list = test_attrs[attr_type]
        layer = best_layers.get(attr_type, best_layers.get("size", n_layers // 2))
        
        H_layer = all_H_dict[attr_type][layer]
        G = H_layer
        G_centered, noun_means = noun_centered_G(G, labels)
        Y = build_indicator(labels, STIMULI[f"{attr_type}_attrs"], attr_type)
        
        pca, cca, P_s, U = compute_signal_subspace(G_centered, Y)
        if pca is None:
            continue
        
        global_signal_ops = {}
        for attr in attr_list:
            op = extract_signal_operator(G_centered, labels, attr, attr_type, pca, P_s)
            if op is not None:
                global_signal_ops[attr] = op
        
        # 名词级f_base: 对每个测试名词, 用它自己的noun_means作为f_base
        # 这与Phase LVI P321不同: P321用的是family_fbase(族平均)
        # 这里直接用noun自身的f_base = noun_means[noun] - 全局均值
        all_noun_mean = np.mean(list(noun_means.values()), axis=0)
        noun_fbase = {n: m - all_noun_mean for n, m in noun_means.items()}
        
        # 5族×12名词×6属性 = 360测试/族
        fam_results = {}
        for fam in FAMILY_NAMES:
            total, success = 0, 0
            
            for test_noun in STIMULI[fam]:
                h_noun_data = noun_means.get(test_noun)
                if h_noun_data is None:
                    continue
                
                # 名词级f_base: 用该名词自己的偏移
                f_base_noun = noun_fbase.get(test_noun, np.zeros(D))
                
                for test_attr in attr_list:
                    if test_attr not in global_signal_ops:
                        continue
                    
                    # 用Phase LVI发现的最优α,β
                    if "qwen" in model_name:
                        alpha, beta = 0.5, 0.5
                    elif "deepseek" in model_name:
                        alpha, beta = 3.0, 0.5
                    else:
                        alpha, beta = 0.5, 5.0
                    
                    logits_new, logits_diff = intervene_with_operator(
                        model, device, h_noun_data, global_signal_ops[test_attr],
                        layer, f_base=f_base_noun, alpha=alpha, beta=beta)
                    
                    hit, rank, logit_change = check_intervention_success(
                        logits_new, logits_diff, tokenizer, test_attr)
                    
                    if hit:
                        success += 1
                    total += 1
            
            rate = success / total * 100 if total > 0 else 0
            fam_results[fam] = {"success": success, "total": total, "rate": rate}
            L.log(f"  {attr_type} {fam[:6]}(noun-fbase): {success}/{total} = {rate:.1f}%")
        
        results[attr_type] = fam_results
    
    return results

# ===================== P323: 大规模β扫描 =====================
def run_p323(model, tokenizer, device, all_H_dict, n_layers, D, labels_dict, best_layers):
    """大规模β扫描: α∈{0.5,1,2,3,5}, β∈{0.1,0.3,0.5,1,2,3,5,8,10,15,20}"""
    L.log("=== P323: 大规模β扫描 ===")
    
    results = {}
    test_attrs = {
        "color": STIMULI["color_attrs"][:6],
        "taste": STIMULI["taste_attrs"][:6],
        "size": STIMULI["size_attrs"][:6],
    }
    
    # 测试名词: 每族4个
    test_nouns = []
    for fam in FAMILY_NAMES:
        test_nouns.extend(STIMULI[fam][::3][:4])
    
    alphas = [0.5, 1.0, 2.0, 3.0, 5.0]
    betas = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]
    
    for attr_type in ["color", "taste", "size"]:
        labels = labels_dict[attr_type]
        attr_list = test_attrs[attr_type]
        layer = best_layers.get(attr_type, best_layers.get("size", n_layers // 2))
        
        H_layer = all_H_dict[attr_type][layer]
        G = H_layer
        G_centered, noun_means = noun_centered_G(G, labels)
        Y = build_indicator(labels, STIMULI[f"{attr_type}_attrs"], attr_type)
        
        pca, cca, P_s, U = compute_signal_subspace(G_centered, Y)
        if pca is None:
            continue
        
        global_signal_ops = {}
        for attr in attr_list:
            op = extract_signal_operator(G_centered, labels, attr, attr_type, pca, P_s)
            if op is not None:
                global_signal_ops[attr] = op
        
        all_noun_mean = np.mean(list(noun_means.values()), axis=0)
        noun_fbase = {n: m - all_noun_mean for n, m in noun_means.items()}
        
        attr_results = {}
        for alpha in alphas:
            for beta in betas:
                total, success = 0, 0
                for test_noun in test_nouns:
                    h_noun_data = noun_means.get(test_noun)
                    if h_noun_data is None:
                        continue
                    f_base_noun = noun_fbase.get(test_noun, np.zeros(D))
                    
                    for test_attr in attr_list:
                        if test_attr not in global_signal_ops:
                            continue
                        
                        logits_new, logits_diff = intervene_with_operator(
                            model, device, h_noun_data, global_signal_ops[test_attr],
                            layer, f_base=f_base_noun, alpha=alpha, beta=beta)
                        
                        hit, rank, logit_change = check_intervention_success(
                            logits_new, logits_diff, tokenizer, test_attr)
                        
                        if hit:
                            success += 1
                        total += 1
                
                rate = success / total * 100 if total > 0 else 0
                key = f"a{alpha}_b{beta}"
                attr_results[key] = {"alpha": alpha, "beta": beta, "rate": rate, "success": success, "total": total}
        
        # 找最优
        best_key = max(attr_results, key=lambda k: attr_results[k]["rate"])
        best_r = attr_results[best_key]
        L.log(f"  {attr_type} 最优: α={best_r['alpha']}, β={best_r['beta']}, rate={best_r['rate']:.1f}%")
        
        # 只保留top-5
        top5 = sorted(attr_results.items(), key=lambda x: x[1]["rate"], reverse=True)[:5]
        results[attr_type] = {"best": best_r, "top5": {k: v for k, v in top5}}
    
    return results

# ===================== P324: 全层+β联合扫描 =====================
def run_p324(model, tokenizer, device, all_H_dict, n_layers, D, labels_dict):
    """全层+β联合扫描: 每层扫描β∈{1,3,5,10,20}"""
    L.log("=== P324: 全层+β联合扫描(大小属性) ===")
    
    results = {}
    attr_type = "size"
    attr_list = STIMULI["size_attrs"][:6]
    labels = labels_dict[attr_type]
    
    betas = [1.0, 3.0, 5.0, 10.0, 20.0]
    alpha = 1.0  # 固定α
    
    test_nouns = []
    for fam in FAMILY_NAMES:
        test_nouns.extend(STIMULI[fam][::3][:4])
    
    layer_results = {}
    for layer_idx in range(1, n_layers, 2):
        H_layer = all_H_dict[attr_type][layer_idx]
        G = H_layer
        G_centered, noun_means = noun_centered_G(G, labels)
        Y = build_indicator(labels, STIMULI[f"{attr_type}_attrs"], attr_type)
        
        pca, cca, P_s, U = compute_signal_subspace(G_centered, Y)
        if pca is None:
            continue
        
        global_signal_ops = {}
        for attr in attr_list:
            op = extract_signal_operator(G_centered, labels, attr, attr_type, pca, P_s)
            if op is not None:
                global_signal_ops[attr] = op
        
        all_noun_mean = np.mean(list(noun_means.values()), axis=0)
        noun_fbase = {n: m - all_noun_mean for n, m in noun_means.items()}
        
        best_beta_for_layer = None
        best_rate = 0
        
        for beta in betas:
            total, success = 0, 0
            for test_noun in test_nouns:
                h_noun_data = noun_means.get(test_noun)
                if h_noun_data is None:
                    continue
                f_base_noun = noun_fbase.get(test_noun, np.zeros(D))
                
                for test_attr in attr_list:
                    if test_attr not in global_signal_ops:
                        continue
                    
                    logits_new, logits_diff = intervene_with_operator(
                        model, device, h_noun_data, global_signal_ops[test_attr],
                        layer_idx, f_base=f_base_noun, alpha=alpha, beta=beta)
                    
                    hit, rank, logit_change = check_intervention_success(
                        logits_new, logits_diff, tokenizer, test_attr)
                    
                    if hit:
                        success += 1
                    total += 1
            
            rate = success / total * 100 if total > 0 else 0
            if rate > best_rate:
                best_rate = rate
                best_beta_for_layer = beta
        
        if best_beta_for_layer is not None:
            layer_results[layer_idx] = {"best_beta": best_beta_for_layer, "best_rate": best_rate}
            if best_rate > 0:
                L.log(f"  L{layer_idx}: β={best_beta_for_layer}, rate={best_rate:.1f}%")
    
    # 找全局最优
    if layer_results:
        best_layer = max(layer_results, key=lambda l: layer_results[l]["best_rate"])
        best_info = layer_results[best_layer]
        L.log(f"  全局最优: L{best_layer}, β={best_info['best_beta']}, rate={best_info['best_rate']:.1f}%")
        results["best"] = {"layer": best_layer, "beta": best_info["best_beta"], "rate": best_info["best_rate"]}
        results["layers"] = layer_results
    
    return results

# ===================== P325: 12属性完整测试 =====================
def run_p325(model, tokenizer, device, all_H_dict, n_layers, D, labels_dict, best_layers, best_params):
    """12属性完整测试: 验证干预覆盖面"""
    L.log("=== P325: 12属性完整测试 ===")
    
    results = {}
    
    for attr_type in ["color", "taste", "size"]:
        labels = labels_dict[attr_type]
        attr_list = STIMULI[f"{attr_type}_attrs"]  # 全部12个属性
        layer = best_layers.get(attr_type, best_layers.get("size", n_layers // 2))
        alpha = best_params.get(attr_type, {}).get("alpha", 1.0)
        beta = best_params.get(attr_type, {}).get("beta", 5.0)
        
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
        
        # 每个属性单独统计
        attr_stats = {}
        for test_attr in attr_list:
            if test_attr not in global_signal_ops:
                continue
            
            total, success = 0, 0
            for fam in FAMILY_NAMES:
                for test_noun in STIMULI[fam]:
                    h_noun_data = noun_means.get(test_noun)
                    if h_noun_data is None:
                        continue
                    f_base_noun = noun_fbase.get(test_noun, np.zeros(D))
                    
                    logits_new, logits_diff = intervene_with_operator(
                        model, device, h_noun_data, global_signal_ops[test_attr],
                        layer, f_base=f_base_noun, alpha=alpha, beta=beta)
                    
                    hit, rank, logit_change = check_intervention_success(
                        logits_new, logits_diff, tokenizer, test_attr)
                    
                    if hit:
                        success += 1
                    total += 1
            
            rate = success / total * 100 if total > 0 else 0
            attr_stats[test_attr] = {"success": success, "total": total, "rate": rate}
        
        # 汇总
        total_all = sum(s["total"] for s in attr_stats.values())
        success_all = sum(s["success"] for s in attr_stats.values())
        avg_rate = success_all / total_all * 100 if total_all > 0 else 0
        
        L.log(f"  {attr_type}(12属性): {success_all}/{total_all} = {avg_rate:.1f}%")
        for attr, stat in sorted(attr_stats.items(), key=lambda x: x[1]["rate"], reverse=True)[:4]:
            L.log(f"    {attr}: {stat['rate']:.1f}%")
        
        results[attr_type] = {"avg_rate": avg_rate, "attr_stats": attr_stats}
    
    return results

# ===================== 主函数 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    model_name = args.model
    L.log(f"=== Phase LVII: 名词级f_base + 大规模β扫描 — {model_name} ===")
    
    # 加载模型
    L.log(f"加载模型: {model_name}")
    model, tokenizer, device = load_model(model_name)
    n_layers = model.config.num_hidden_layers + 1
    D = model.config.hidden_size
    L.log(f"模型: {model_name}, 层数={n_layers}, 维度={D}")
    
    # Phase LVI发现的最优层
    best_layers_map = {
        "qwen3": {"color": 33, "taste": 9, "size": 9},
        "glm4": {"color": 37, "taste": 39, "size": 33},
        "deepseek7b": {"color": 9, "taste": 9, "size": 9},
    }
    best_layers = best_layers_map.get(model_name, {"color": n_layers//2, "taste": n_layers//2, "size": n_layers//2})
    
    # Phase LVI发现的最优参数
    best_params_map = {
        "qwen3": {"color": {"alpha": 0.5, "beta": 0.5}, "taste": {"alpha": 0.5, "beta": 0.5}, "size": {"alpha": 0.5, "beta": 0.5}},
        "glm4": {"color": {"alpha": 0.5, "beta": 5.0}, "taste": {"alpha": 0.5, "beta": 5.0}, "size": {"alpha": 0.5, "beta": 5.0}},
        "deepseek7b": {"color": {"alpha": 1.5, "beta": 0.5}, "taste": {"alpha": 0.5, "beta": 0.5}, "size": {"alpha": 3.0, "beta": 0.5}},
    }
    best_params = best_params_map.get(model_name, {})
    
    # 采集隐藏状态(所有层)
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
    
    # ========== P322: 名词级f_base跨族泛化 ==========
    L.log("P322: 名词级f_base跨族泛化...")
    p322_results = run_p322(model, tokenizer, device, all_H_dict, n_layers, D, labels_dict, best_layers, model_name)
    
    # ========== P323: 大规模β扫描 ==========
    L.log("P323: 大规模β扫描...")
    p323_results = run_p323(model, tokenizer, device, all_H_dict, n_layers, D, labels_dict, best_layers)
    
    # ========== P324: 全层+β联合扫描 ==========
    L.log("P324: 全层+β联合扫描(大小属性)...")
    p324_results = run_p324(model, tokenizer, device, all_H_dict, n_layers, D, labels_dict)
    
    # ========== P325: 12属性完整测试 ==========
    L.log("P325: 12属性完整测试...")
    p325_results = run_p325(model, tokenizer, device, all_H_dict, n_layers, D, labels_dict, best_layers, best_params)
    
    # ========== 保存结果 ==========
    all_results = {
        "model": model_name,
        "n_layers": n_layers,
        "D": D,
        "best_layers": best_layers,
        "best_params": best_params,
        "p322_noun_fbase": p322_results,
        "p323_large_beta": p323_results,
        "p324_layer_beta": p324_results,
        "p325_12attrs": p325_results,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M"),
    }
    
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = OUT_DIR / f"phase_lvii_p322_325_{model_name}_{ts}.json"
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
