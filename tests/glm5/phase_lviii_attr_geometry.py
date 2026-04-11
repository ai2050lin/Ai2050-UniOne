"""
Phase LVIII-P326/327/328/329/330: 细粒度属性几何分析
======================================================================

Phase LVII核心突破:
  1. GLM4: 7个属性100%干预! (short/long/heavy/light/sweet/soft/fresh)
  2. 反义词对最易干预: short↔long, heavy↔light
  3. GLM4颜色β=8.0最优, DS7B α=0.5/β=1.0(P323)
  4. 名词级f_base有效: GLM4大小50%
  5. DS7B P325大小0%: 参数不当

Phase LVIII核心目标: 理解为什么某些属性100%而其他0%

五大实验:
  P326: 反义词编码几何分析
    - 计算反义词对的信号算子之间的余弦相似度
    - 分析short↔long, heavy↔light, big↔small等
    - 假设: 反义词的信号算子方向相反(余弦≈-1)
    - 12个反义词/近义词对 × 3属性类型

  P327: 属性词频vs干预成功率
    - 统计每个属性词在训练语料中的词频
    - 分析词频是否与干预成功率相关
    - 使用模型tokenizer的词频统计

  P328: 信号算子范数vs干预成功率
    - 计算每个属性信号算子的L2范数
    - 分析范数是否与成功率相关
    - 假设: 范数越大→信号越强→成功率越高

  P329: 逐属性最优β扫描
    - 对每个属性单独扫描β∈{0.1,0.3,0.5,1,2,3,5,8,10,15,20,30,50}
    - 找到每个属性的最优β
    - 目标: 找到为什么某些属性需要大β

  P330: DS7B参数修复+逐属性最优
    - 用P323发现的最优参数(α=0.5, β=1.0)
    - 再对每个属性单独扫描β
    - 目标: 验证DS7B是否也能达到高成功率

数据规模: 12属性×60名词×13β值 × 3模型
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
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import functools, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
print = functools.partial(print, flush=True)

OUT_DIR = Path("tests/glm5_temp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUT_DIR / "phase_lviii_log.txt"

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

# 反义词/近义词对定义
ANTONYM_PAIRS = {
    "size": [
        ("big", "small"), ("tall", "short"), ("long", "short"),
        ("heavy", "light"), ("thick", "thin"), ("wide", "narrow"),  # narrow不在列表中
        ("huge", "tiny"),
    ],
    "color": [
        ("white", "black"), ("red", "green"),  # 互补色
        ("gray", "gold"),  # 对比色
    ],
    "taste": [
        ("sweet", "sour"), ("sweet", "bitter"), ("salty", "mild"),
        ("crisp", "soft"), ("spicy", "mild"), ("fresh", "rich"),
    ],
}

def noun_centered_G(G, labels):
    """去名词均值"""
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

# ===================== P326: 反义词编码几何 =====================
def run_p326(model, tokenizer, device, all_H_dict, n_layers, D, labels_dict, best_layers):
    """反义词编码几何分析: 反义词信号算子的余弦相似度"""
    L.log("=== P326: 反义词编码几何分析 ===")
    
    results = {}
    
    for attr_type in ["size", "taste", "color"]:
        labels = labels_dict[attr_type]
        attr_list = STIMULI[f"{attr_type}_attrs"]
        layer = best_layers.get(attr_type, best_layers.get("size", n_layers // 2))
        
        H_layer = all_H_dict[attr_type][layer]
        G = H_layer
        G_centered, noun_means = noun_centered_G(G, labels)
        Y = build_indicator(labels, attr_list, attr_type)
        
        pca, cca, P_s, U = compute_signal_subspace(G_centered, Y, n_cca=min(10, len(attr_list)-1))
        if pca is None:
            L.log(f"  {attr_type}: PCA/CCA失败, 跳过")
            continue
        
        # 提取所有属性的信号算子
        signal_ops = {}
        for attr in attr_list:
            op = extract_signal_operator(G_centered, labels, attr, attr_type, pca, P_s)
            if op is not None:
                signal_ops[attr] = op
        
        # 计算信号算子范数
        op_norms = {attr: np.linalg.norm(op) for attr, op in signal_ops.items()}
        L.log(f"  {attr_type} 信号算子范数: " + 
              ", ".join(f"{a}={n:.4f}" for a, n in sorted(op_norms.items(), key=lambda x: -x[1])[:5]))
        
        # 计算反义词对的余弦相似度
        pair_results = []
        for a1, a2 in ANTONYM_PAIRS.get(attr_type, []):
            if a1 in signal_ops and a2 in signal_ops:
                cos_sim = 1 - cosine(signal_ops[a1], signal_ops[a2])
                pair_results.append({
                    "pair": f"{a1}↔{a2}",
                    "cosine_sim": float(cos_sim),
                    "norm1": float(op_norms[a1]),
                    "norm2": float(op_norms[a2]),
                })
                L.log(f"  {a1}↔{a2}: 余弦相似度={cos_sim:.4f}, 范数=({op_norms[a1]:.4f}, {op_norms[a2]:.4f})")
        
        # 计算所有属性之间的余弦相似度矩阵
        n_attrs = len(attr_list)
        cos_matrix = np.zeros((n_attrs, n_attrs))
        for i, a1 in enumerate(attr_list):
            for j, a2 in enumerate(attr_list):
                if a1 in signal_ops and a2 in signal_ops:
                    cos_matrix[i, j] = 1 - cosine(signal_ops[a1], signal_ops[a2])
        
        results[attr_type] = {
            "pair_results": pair_results,
            "op_norms": op_norms,
            "cos_matrix_attrs": attr_list,
            "cos_matrix": cos_matrix.tolist(),
        }
    
    return results

# ===================== P327: 属性词频vs成功率 =====================
def run_p327(tokenizer, p325_results_if_any=None):
    """属性词频vs干预成功率分析"""
    L.log("=== P327: 属性词频vs干预成功率 ===")
    
    results = {}
    
    for attr_type in ["color", "taste", "size"]:
        attr_list = STIMULI[f"{attr_type}_attrs"]
        
        # 计算每个属性词的token数量和词频估计
        attr_token_info = {}
        for attr in attr_list:
            tok_ids = tokenizer.encode(attr, add_special_tokens=False)
            # 词频估计: 单token词更常见, 多token词更罕见
            n_tokens = len(tok_ids)
            attr_token_info[attr] = {
                "n_tokens": n_tokens,
                "token_ids": tok_ids,
            }
            L.log(f"  {attr}: {n_tokens} tokens = {tok_ids}")
        
        results[attr_type] = attr_token_info
    
    return results

# ===================== P328: 信号算子范数vs成功率 =====================
def run_p328(p326_results, phase_lvii_results_path=None):
    """信号算子范数vs干预成功率"""
    L.log("=== P328: 信号算子范数vs干预成功率 ===")
    
    results = {}
    
    # 读取Phase LVII的结果
    phase_lvii_data = None
    if phase_lvii_results_path and os.path.exists(phase_lvii_results_path):
        with open(phase_lvii_results_path, 'r', encoding='utf-8') as f:
            phase_lvii_data = json.load(f)
    
    for attr_type in ["color", "taste", "size"]:
        if attr_type not in p326_results:
            continue
        
        op_norms = p326_results[attr_type]["op_norms"]
        attr_list = STIMULI[f"{attr_type}_attrs"]
        
        # 从Phase LVII获取成功率
        success_rates = {}
        if phase_lvii_data and "p325_12attrs" in phase_lvii_data:
            p325_data = phase_lvii_data["p325_12attrs"]
            if attr_type in p325_data and "attr_stats" in p325_data[attr_type]:
                for attr, stat in p325_data[attr_type]["attr_stats"].items():
                    success_rates[attr] = stat["rate"]
        
        # 合并数据
        combined = []
        for attr in attr_list:
            norm = op_norms.get(attr, 0)
            rate = success_rates.get(attr, 0)
            combined.append({"attr": attr, "norm": float(norm), "rate": float(rate)})
            L.log(f"  {attr}: 范数={norm:.4f}, 成功率={rate:.1f}%")
        
        # 计算相关系数
        norms = [c["norm"] for c in combined]
        rates = [c["rate"] for c in combined]
        if len(norms) > 2 and np.std(norms) > 0 and np.std(rates) > 0:
            corr, pval = stats.pearsonr(norms, rates)
            L.log(f"  {attr_type}: 范数-成功率 Pearson r={corr:.4f}, p={pval:.4f}")
        else:
            corr, pval = 0, 1
        
        results[attr_type] = {
            "combined": combined,
            "pearson_r": float(corr),
            "p_value": float(pval),
        }
    
    return results

# ===================== P329: 逐属性最优β扫描 =====================
def run_p329(model, tokenizer, device, all_H_dict, n_layers, D, labels_dict, best_layers, model_name):
    """逐属性最优β扫描: 每个属性找自己的最优β"""
    L.log("=== P329: 逐属性最优β扫描 ===")
    
    beta_range = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0]
    alpha_fixed = 0.5  # 固定α=0.5
    
    results = {}
    
    for attr_type in ["size", "taste", "color"]:
        labels = labels_dict[attr_type]
        attr_list = STIMULI[f"{attr_type}_attrs"]
        layer = best_layers.get(attr_type, best_layers.get("size", n_layers // 2))
        
        H_layer = all_H_dict[attr_type][layer]
        G = H_layer
        G_centered, noun_means = noun_centered_G(G, labels)
        Y = build_indicator(labels, attr_list, attr_type)
        
        pca, cca, P_s, U = compute_signal_subspace(G_centered, Y, n_cca=min(10, len(attr_list)-1))
        if pca is None:
            continue
        
        # 提取所有信号算子
        global_signal_ops = {}
        for attr in attr_list:
            op = extract_signal_operator(G_centered, labels, attr, attr_type, pca, P_s)
            if op is not None:
                global_signal_ops[attr] = op
        
        all_noun_mean = np.mean(list(noun_means.values()), axis=0)
        noun_fbase = {n: m - all_noun_mean for n, m in noun_means.items()}
        
        # 逐属性扫描β
        attr_best_beta = {}
        for test_attr in attr_list:
            if test_attr not in global_signal_ops:
                continue
            
            best_beta = 1.0
            best_rate = 0.0
            beta_curve = []
            
            for beta in beta_range:
                total, success = 0, 0
                # 用5族各3个名词 = 15测试(加速)
                for fam in FAMILY_NAMES:
                    for test_noun in STIMULI[fam][:3]:
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
                beta_curve.append({"beta": beta, "rate": rate})
                
                if rate > best_rate:
                    best_rate = rate
                    best_beta = beta
            
            attr_best_beta[test_attr] = {
                "best_beta": best_beta,
                "best_rate": best_rate,
                "beta_curve": beta_curve,
            }
            L.log(f"  {test_attr}: 最优β={best_beta}, 成功率={best_rate:.1f}%")
        
        results[attr_type] = attr_best_beta
    
    return results

# ===================== P330: DS7B参数修复重测 =====================
def run_p330(model, tokenizer, device, all_H_dict, n_layers, D, labels_dict, best_layers, model_name):
    """DS7B参数修复: 用P323的最优参数重测"""
    L.log("=== P330: DS7B参数修复重测 ===")
    
    if "deepseek" not in model_name:
        L.log("  跳过(非DS7B模型)")
        return None
    
    # P323发现的最优参数: α=0.5, β=1.0
    # 再加上逐属性扫描
    beta_range = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]
    
    results = {}
    
    for attr_type in ["size", "taste", "color"]:
        labels = labels_dict[attr_type]
        attr_list = STIMULI[f"{attr_type}_attrs"]
        layer = best_layers.get(attr_type, 9)
        
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
        
        # 全β扫描×全属性
        attr_beta_results = {}
        for test_attr in attr_list:
            if test_attr not in global_signal_ops:
                continue
            
            best_beta = 1.0
            best_rate = 0.0
            
            for beta in beta_range:
                total, success = 0, 0
                for fam in FAMILY_NAMES:
                    for test_noun in STIMULI[fam]:
                        h_noun_data = noun_means.get(test_noun)
                        if h_noun_data is None:
                            continue
                        f_base_noun = noun_fbase.get(test_noun, np.zeros(D))
                        
                        logits_new, logits_diff = intervene_with_operator(
                            model, device, h_noun_data, global_signal_ops[test_attr],
                            layer, f_base=f_base_noun, alpha=0.5, beta=beta)
                        
                        hit, rank, logit_change = check_intervention_success(
                            logits_new, logits_diff, tokenizer, test_attr)
                        
                        if hit:
                            success += 1
                        total += 1
                
                rate = success / total * 100 if total > 0 else 0
                
                if rate > best_rate:
                    best_rate = rate
                    best_beta = beta
            
            attr_beta_results[test_attr] = {"best_beta": best_beta, "best_rate": best_rate}
            L.log(f"  {test_attr}: 最优β={best_beta}, 成功率={best_rate:.1f}%")
        
        results[attr_type] = attr_beta_results
    
    return results

# ===================== 主函数 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    model_name = args.model
    L.log(f"=== Phase LVIII: 细粒度属性几何分析 — {model_name} ===")
    
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
    
    # ========== P326: 反义词编码几何 ==========
    L.log("P326: 反义词编码几何分析...")
    p326_results = run_p326(model, tokenizer, device, all_H_dict, n_layers, D, labels_dict, best_layers)
    
    # ========== P327: 属性词频 ==========
    L.log("P327: 属性词频分析...")
    # 查找Phase LVII结果
    phase_lvii_path = None
    for f in sorted(OUT_DIR.glob(f"phase_lvii_p322_325_{model_name}_*.json"), reverse=True)[:1]:
        phase_lvii_path = str(f)
    p327_results = run_p327(tokenizer, phase_lvii_path)
    
    # ========== P328: 信号算子范数vs成功率 ==========
    L.log("P328: 信号算子范数vs成功率...")
    p328_results = run_p328(p326_results, phase_lvii_path)
    
    # ========== P329: 逐属性最优β扫描 ==========
    L.log("P329: 逐属性最优β扫描...")
    p329_results = run_p329(model, tokenizer, device, all_H_dict, n_layers, D, labels_dict, best_layers, model_name)
    
    # ========== P330: DS7B参数修复 ==========
    L.log("P330: DS7B参数修复...")
    p330_results = run_p330(model, tokenizer, device, all_H_dict, n_layers, D, labels_dict, best_layers, model_name)
    
    # ========== 保存结果 ==========
    all_results = {
        "model": model_name,
        "n_layers": n_layers,
        "D": D,
        "best_layers": best_layers,
        "p326_antonym_geometry": p326_results,
        "p327_token_freq": p327_results,
        "p328_norm_vs_rate": p328_results,
        "p329_per_attr_beta": p329_results,
        "p330_ds7b_fix": p330_results,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M"),
    }
    
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = OUT_DIR / f"phase_lviii_p326_330_{model_name}_{ts}.json"
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
