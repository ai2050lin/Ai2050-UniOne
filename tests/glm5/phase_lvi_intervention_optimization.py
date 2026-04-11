"""
Phase LVI-P318/319/320/321: 干预策略优化
======================================================================

Phase LV核心结论:
  1. fbase+global_signal最优: GLM4大小66.7%, DS7B大小50%
  2. f_base是语义锚点, 不是噪声
  3. 大小>味道>颜色: 物理属性最可控
  4. Qwen3干预0%: 架构原因待查
  5. 末层不如前层: L36>L39(GLM4), L24>L27(DS7B)

Phase LVI核心目标: 优化干预策略, 突破66.7%天花板

四大实验:
  P318: 全层扫描干预成功率 — 找到最优干预层
    - 对fbase+global方法在所有层做干预
    - 每层测试5族×12名词×3属性 = 180个干预
    - 目标: 找到干预成功率最高的层, 预期中间层更有效

  P319: 信号强度缩放优化 — α·f_base + β·signal
    - 固定最优层, 扫描α∈{0.5,1,1.5,2,3,5}, β∈{0.5,1,1.5,2,3,5}
    - 每组参数测试5族×12名词×3属性 = 180个干预
    - 目标: 找到最优α,β组合, 预期α=1-2, β=1-3

  P320: 多属性联合干预 — 同时干预2-3个属性
    - 在最优层+最优α,β下, 联合干预颜色+大小, 大小+味道等
    - h_new = h_noun + f_base + β1·signal1 + β2·signal2
    - 目标: 多属性联合干预成功率>50%

  P321: 大规模跨族泛化测试 — 5族×12名词×3属性×全部参数
    - 用最优参数组合做最终的跨族泛化测试
    - 包含5×5族间矩阵 + 失败案例分析
    - 目标: 验证fbase+global的极限, 干预成功率>80%

数据规模: 2160三元组 × 30模板 + 全层扫描 + 参数网格搜索
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

LOG_FILE = OUT_DIR / "phase_lvi_log.txt"

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
    """构建属性indicator矩阵: one-hot编码"""
    N = len(labels)
    K = len(attr_list)
    Y = np.zeros((N, K))
    attr_to_idx = {a: i for i, a in enumerate(attr_list)}
    for i, l in enumerate(labels):
        if l[1] == attr_type:
            Y[i, attr_to_idx[l[2]]] = 1.0
    return Y

def compute_signal_subspace(G_centered, Y, n_cca=10):
    """用CCA计算信号子空间投影矩阵和纯净算子"""
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
    
    U = cca.x_weights_  # (pca_dim, n_cca)
    P_s_pca = U @ np.linalg.inv(U.T @ U + 1e-6 * np.eye(n_cca)) @ U.T
    
    return pca, cca, P_s_pca, U

def extract_signal_operator(G_centered, labels, attr, attr_type, pca, P_s_pca):
    """提取纯净属性算子(在G_centered空间, 反投影回原始空间)"""
    attr_mask = [i for i, l in enumerate(labels) if l[2] == attr]
    if len(attr_mask) < 2:
        return None
    
    G_attr = G_centered[attr_mask]
    G_attr_mean = np.mean(G_attr, axis=0)
    
    # PCA降维 → CCA信号子空间投影 → 反投影
    G_pca = pca.transform(G_attr_mean.reshape(1, -1))
    G_signal_pca = P_s_pca @ G_pca.T
    G_signal = pca.inverse_transform(G_signal_pca.T)[0]
    
    return G_signal

def intervene_with_operator(model, device, h_noun, operator, layer, f_base=None, alpha=1.0, beta=1.0):
    """用算子干预, 支持强度缩放, 含norm层处理
    h_new = h_noun + alpha*f_base + beta*operator
    返回logits_new, logits_diff
    """
    h_new = h_noun.copy()
    if f_base is not None:
        h_new = h_new + alpha * f_base
    h_new = h_new + beta * operator
    
    with torch.no_grad():
        h_tensor = torch.tensor(h_new, dtype=torch.bfloat16).unsqueeze(0).unsqueeze(0).to(device)
        h_base_tensor = torch.tensor(h_noun, dtype=torch.bfloat16).unsqueeze(0).unsqueeze(0).to(device)
        
        # 获取lm_head和norm层
        lm_head = model.lm_head if hasattr(model, 'lm_head') else model.get_output_embeddings()
        
        # 检查是否有norm层(Qwen3/GLM4等都有model.norm)
        norm = None
        if hasattr(model, 'model') and hasattr(model.model, 'norm'):
            norm = model.model.norm
        
        # 先做norm再做lm_head投影
        if norm is not None:
            h_tensor_normed = norm(h_tensor)
            h_base_tensor_normed = norm(h_base_tensor)
            logits_new = lm_head(h_tensor_normed)[0, 0].float().cpu().numpy()
            logits_base = lm_head(h_base_tensor_normed)[0, 0].float().cpu().numpy()
        else:
            logits_new = lm_head(h_tensor)[0, 0].float().cpu().numpy()
            logits_base = lm_head(h_base_tensor)[0, 0].float().cpu().numpy()
    
    logits_diff = logits_new - logits_base
    return logits_new, logits_diff

def check_intervention_success(logits_new, logits_diff, tokenizer, target_attr, top_k=20):
    """检查干预是否成功: logits_diff的top-k中是否包含目标属性词"""
    attr_tok_ids = tokenizer.encode(target_attr, add_special_tokens=False)
    if len(attr_tok_ids) == 0:
        return False, -1, 0.0
    
    attr_tok_id = attr_tok_ids[0]
    
    # 检查logits_diff的top-k
    top_k_ids = np.argsort(logits_diff)[-top_k:][::-1]
    hit = attr_tok_id in top_k_ids
    rank = int(np.where(top_k_ids == attr_tok_id)[0][0]) + 1 if hit else -1
    
    # logit变化值
    target_logit_change = float(logits_diff[attr_tok_id]) if attr_tok_id < len(logits_diff) else 0.0
    
    return hit, rank, target_logit_change

# ===================== 采集隐藏状态 =====================
def collect_hidden_states(model, tokenizer, device, labels, templates, batch_size=16):
    """收集所有三元组在所有层的隐藏状态"""
    N = len(labels)
    n_layers = model.config.num_hidden_layers + 1
    
    # 先用一个样本获取维度
    sample_text = templates[0].format(word=labels[0][0])
    toks = tokenizer(sample_text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(toks.input_ids, output_hidden_states=True)
    D = out.hidden_states[0].shape[-1]
    
    # 为每层预分配
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
    """收集名词的隐藏状态(用于f_base)"""
    noun_h = {}
    for noun in nouns:
        # 用30个模板取平均
        h_list = []
        for t in templates:
            text = t.format(word=noun)
            toks = tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model(toks.input_ids, output_hidden_states=True)
            h_list.append(out.hidden_states[layer][0, -1].float().cpu().numpy())
        noun_h[noun] = np.mean(h_list, axis=0)
    return noun_h

# ===================== P318: 全层扫描 =====================
def run_p318(model, tokenizer, device, all_H, n_layers, D, labels_dict, noun_h_dict):
    """全层扫描干预成功率: 在每一层用fbase+global方法做干预"""
    L.log("=== P318: 全层扫描干预成功率 ===")
    
    results = {}
    test_attrs_short = {
        "color": STIMULI["color_attrs"][:6],
        "taste": STIMULI["taste_attrs"][:6],
        "size": STIMULI["size_attrs"][:6],
    }
    
    # 测试名词: 每族4个(均匀采样)
    test_nouns = {}
    for fam in FAMILY_NAMES:
        test_nouns[fam] = STIMULI[fam][::3][:4]
    
    # 每层做干预
    for layer in range(1, n_layers):  # 跳过第0层(embedding)
        layer_result = {"layer": layer}
        
        for attr_type in ["color", "taste", "size"]:
            labels = labels_dict[attr_type]
            attr_list = test_attrs_short[attr_type]
            
            # 获取该层的G矩阵
            H_layer = all_H[layer - 1] if layer > 0 else all_H[0]
            # 重新采集: 使用当前层的H
            # 但我们用的是预采集的all_H
            G = H_layer
            G_centered, noun_means = noun_centered_G(G, labels)
            Y = build_indicator(labels, STIMULI[f"{attr_type}_attrs"], attr_type)
            
            # 信号子空间
            pca, cca, P_s, U = compute_signal_subspace(G_centered, Y)
            if pca is None:
                continue
            
            # 全局信号算子
            global_signal_ops = {}
            for attr in attr_list:
                op = extract_signal_operator(G_centered, labels, attr, attr_type, pca, P_s)
                if op is not None:
                    global_signal_ops[attr] = op
            
            # f_base: 各族的noun_means平均
            family_fbase = {}
            for fam in FAMILY_NAMES:
                fam_nouns = STIMULI[fam]
                fam_means = {n: noun_means[n] for n in fam_nouns if n in noun_means}
                if fam_means:
                    family_fbase[fam] = np.mean(list(fam_means.values()), axis=0)
            
            # 干预测试
            total, success = 0, 0
            for fam in FAMILY_NAMES:
                for test_noun in test_nouns[fam]:
                    # 获取h_noun
                    h_noun_data = noun_h_dict.get(test_noun)
                    if h_noun_data is None:
                        # 从noun_means获取(同层)
                        h_noun_data = noun_means.get(test_noun)
                    if h_noun_data is None:
                        continue
                    
                    f_base_noun = noun_means.get(test_noun, np.zeros(D))
                    
                    for test_attr in attr_list:
                        if test_attr not in global_signal_ops:
                            continue
                        
                        logits_new, logits_diff = intervene_with_operator(
                            model, device, h_noun_data, global_signal_ops[test_attr],
                            layer, f_base=f_base_noun, alpha=1.0, beta=1.0)
                        
                        hit, rank, logit_change = check_intervention_success(
                            logits_new, logits_diff, tokenizer, test_attr)
                        
                        if hit:
                            success += 1
                        total += 1
            
            rate = success / total * 100 if total > 0 else 0
            layer_result[attr_type] = {"success": success, "total": total, "rate": rate}
            L.log(f"  L{layer} {attr_type}: {success}/{total} = {rate:.1f}%")
        
        results[layer] = layer_result
    
    return results

# ===================== P319: 信号强度缩放 =====================
def run_p319(model, tokenizer, device, all_H, n_layers, D, labels_dict, noun_h_dict, best_layer):
    """信号强度缩放优化: 扫描α和β"""
    L.log(f"=== P319: 信号强度缩放(最优层L{best_layer}) ===")
    
    results = {}
    test_attrs_short = {
        "color": STIMULI["color_attrs"][:6],
        "taste": STIMULI["taste_attrs"][:6],
        "size": STIMULI["size_attrs"][:6],
    }
    
    test_nouns = {}
    for fam in FAMILY_NAMES:
        test_nouns[fam] = STIMULI[fam][::3][:4]
    
    alphas = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    betas = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    layer = best_layer
    
    for attr_type in ["color", "taste", "size"]:
        labels = labels_dict[attr_type]
        attr_list = test_attrs_short[attr_type]
        
        H_layer = all_H[layer]
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
        
        family_fbase = {}
        for fam in FAMILY_NAMES:
            fam_nouns = STIMULI[fam]
            fam_means = {n: noun_means[n] for n in fam_nouns if n in noun_means}
            if fam_means:
                family_fbase[fam] = np.mean(list(fam_means.values()), axis=0)
        
        attr_results = {}
        for alpha in alphas:
            for beta in betas:
                total, success = 0, 0
                for fam in FAMILY_NAMES:
                    for test_noun in test_nouns[fam]:
                        h_noun_data = noun_means.get(test_noun)
                        if h_noun_data is None:
                            continue
                        f_base_noun = noun_means.get(test_noun, np.zeros(D))
                        
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
                key = f"alpha={alpha}_beta={beta}"
                attr_results[key] = {"alpha": alpha, "beta": beta, "success": success, "total": total, "rate": rate}
                L.log(f"  {attr_type} α={alpha} β={beta}: {success}/{total} = {rate:.1f}%")
        
        # 找最优α,β
        best_key = max(attr_results, key=lambda k: attr_results[k]["rate"])
        best_result = attr_results[best_key]
        L.log(f"  {attr_type} 最优: α={best_result['alpha']}, β={best_result['beta']}, rate={best_result['rate']:.1f}%")
        
        results[attr_type] = attr_results
    
    return results

# ===================== P320: 多属性联合干预 =====================
def run_p320(model, tokenizer, device, all_H, n_layers, D, labels_dict, noun_h_dict, best_layer, best_alpha, best_beta):
    """多属性联合干预: 同时干预2-3个属性"""
    L.log(f"=== P320: 多属性联合干预(L{best_layer}, α={best_alpha}, β={best_beta}) ===")
    
    results = {}
    test_nouns = []
    for fam in FAMILY_NAMES:
        test_nouns.extend(STIMULI[fam][::3][:4])  # 20个测试名词
    
    # 组合对
    attr_pairs = [
        ("color", "size"),
        ("color", "taste"),
        ("size", "taste"),
        ("color", "size", "taste"),  # 三属性联合
    ]
    
    layer = best_layer
    
    # 预计算每类属性的信号算子
    signal_ops_all = {}
    for attr_type in ["color", "taste", "size"]:
        labels = labels_dict[attr_type]
        attr_list = STIMULI[f"{attr_type}_attrs"][:6]
        
        H_layer = all_H[layer]
        G = H_layer
        G_centered, noun_means = noun_centered_G(G, labels)
        Y = build_indicator(labels, STIMULI[f"{attr_type}_attrs"], attr_type)
        
        pca, cca, P_s, U = compute_signal_subspace(G_centered, Y)
        if pca is None:
            continue
        
        signal_ops = {}
        for attr in attr_list:
            op = extract_signal_operator(G_centered, labels, attr, attr_type, pca, P_s)
            if op is not None:
                signal_ops[attr] = (op, noun_means)
        
        signal_ops_all[attr_type] = signal_ops
    
    # 对每个组合做联合干预
    for pair in attr_pairs:
        pair_name = "+".join(pair)
        pair_result = {"pair": pair_name, "details": []}
        
        total, success_any, success_all = 0, 0, 0
        
        for test_noun in test_nouns:
            # 获取h_noun(从第一个属性类型的noun_means)
            first_type = pair[0]
            if first_type not in signal_ops_all or not signal_ops_all[first_type]:
                continue
            first_ops = signal_ops_all[first_type]
            first_attr = list(first_ops.keys())[0]
            _, noun_means = first_ops[first_attr]
            
            h_noun_data = noun_means.get(test_noun)
            if h_noun_data is None:
                continue
            f_base_noun = noun_means.get(test_noun, np.zeros(D))
            
            # 选择每个属性类型的一个属性
            for attr_combo in _iter_attr_combos(pair, signal_ops_all):
                # h_new = h_noun + α·f_base + β·(signal1 + signal2 + ...)
                combined_operator = np.zeros(D)
                target_attrs = []
                for attr_type, attr in attr_combo:
                    if attr_type in signal_ops_all and attr in signal_ops_all[attr_type]:
                        op, _ = signal_ops_all[attr_type][attr]
                        combined_operator += op
                        target_attrs.append((attr_type, attr))
                
                if len(target_attrs) < len(pair):
                    continue
                
                logits_new, logits_diff = intervene_with_operator(
                    model, device, h_noun_data, combined_operator,
                    layer, f_base=f_base_noun, alpha=best_alpha, beta=best_beta)
                
                # 检查每个目标属性是否成功
                hits = []
                for attr_type, attr in target_attrs:
                    hit, rank, logit_change = check_intervention_success(
                        logits_new, logits_diff, tokenizer, attr)
                    hits.append(hit)
                
                if any(hits):
                    success_any += 1
                if all(hits):
                    success_all += 1
                total += 1
        
        rate_any = success_any / total * 100 if total > 0 else 0
        rate_all = success_all / total * 100 if total > 0 else 0
        pair_result["success_any"] = success_any
        pair_result["success_all"] = success_all
        pair_result["total"] = total
        pair_result["rate_any"] = rate_any
        pair_result["rate_all"] = rate_all
        L.log(f"  {pair_name}: any={rate_any:.1f}%, all={rate_all:.1f}% ({success_any}/{success_all}/{total})")
        
        results[pair_name] = pair_result
    
    return results

def _iter_attr_combos(pair, signal_ops_all, max_per_type=2):
    """生成属性组合迭代器(每类取前max_per_type个)"""
    from itertools import product
    attrs_per_type = []
    for attr_type in pair:
        if attr_type in signal_ops_all:
            attrs = list(signal_ops_all[attr_type].keys())[:max_per_type]
            attrs_per_type.append([(attr_type, a) for a in attrs])
        else:
            return []
    
    for combo in product(*attrs_per_type):
        yield combo

# ===================== P321: 大规模跨族泛化 =====================
def run_p321(model, tokenizer, device, all_H, n_layers, D, labels_dict, noun_h_dict, best_layer, best_alpha, best_beta):
    """大规模跨族泛化测试: 5×5族间矩阵 + 失败分析"""
    L.log(f"=== P321: 大规模跨族泛化(L{best_layer}, α={best_alpha}, β={best_beta}) ===")
    
    results = {}
    test_attrs_short = {
        "color": STIMULI["color_attrs"][:6],
        "taste": STIMULI["taste_attrs"][:6],
        "size": STIMULI["size_attrs"][:6],
    }
    
    layer = best_layer
    
    for attr_type in ["color", "taste", "size"]:
        labels = labels_dict[attr_type]
        attr_list = test_attrs_short[attr_type]
        
        H_layer = all_H[layer]
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
        
        # 各族f_base
        family_fbase = {}
        for fam in FAMILY_NAMES:
            fam_nouns = STIMULI[fam]
            fam_means = {n: noun_means[n] for n in fam_nouns if n in noun_means}
            if fam_means:
                family_fbase[fam] = np.mean(list(fam_means.values()), axis=0)
        
        # 5×5跨族矩阵(用fbase+global方法)
        matrix = {}
        for source_fam in FAMILY_NAMES:
            for target_fam in FAMILY_NAMES:
                total, success = 0, 0
                failed_cases = []
                
                # 测试名词: 目标族的全部12个
                for test_noun in STIMULI[target_fam]:
                    h_noun_data = noun_means.get(test_noun)
                    if h_noun_data is None:
                        continue
                    
                    # 使用目标族的f_base
                    f_base_target = family_fbase.get(target_fam, np.zeros(D))
                    
                    for test_attr in attr_list:
                        if test_attr not in global_signal_ops:
                            continue
                        
                        logits_new, logits_diff = intervene_with_operator(
                            model, device, h_noun_data, global_signal_ops[test_attr],
                            layer, f_base=f_base_target, alpha=best_alpha, beta=best_beta)
                        
                        hit, rank, logit_change = check_intervention_success(
                            logits_new, logits_diff, tokenizer, test_attr)
                        
                        if hit:
                            success += 1
                        else:
                            failed_cases.append({
                                "noun": test_noun, "attr": test_attr,
                                "logit_change": logit_change,
                                "source_fam": source_fam, "target_fam": target_fam
                            })
                        total += 1
                
                rate = success / total * 100 if total > 0 else 0
                matrix[f"{source_fam}->{target_fam}"] = {
                    "success": success, "total": total, "rate": rate,
                    "failed_cases": failed_cases[:5]  # 只保留前5个失败案例
                }
                L.log(f"  {attr_type} {source_fam[:6]}→{target_fam[:6]}: {success}/{total} = {rate:.1f}%")
        
        # 汇总
        in_family_rates = [matrix[f"{f}->{f}"]["rate"] for f in FAMILY_NAMES if f"{f}->{f}" in matrix]
        cross_family_rates = [matrix[f"{s}->{t}"]["rate"] for s in FAMILY_NAMES for t in FAMILY_NAMES if s != t and f"{s}->{t}" in matrix]
        
        results[attr_type] = {
            "matrix": matrix,
            "in_family_avg": np.mean(in_family_rates) if in_family_rates else 0,
            "cross_family_avg": np.mean(cross_family_rates) if cross_family_rates else 0,
        }
        L.log(f"  {attr_type} 族内avg={np.mean(in_family_rates):.1f}%, 跨族avg={np.mean(cross_family_rates):.1f}%")
    
    return results

# ===================== 主函数 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    model_name = args.model
    L.log(f"=== Phase LVI: 干预策略优化 — {model_name} ===")
    
    # 加载模型
    L.log(f"加载模型: {model_name}")
    model, tokenizer, device = load_model(model_name)
    n_layers = model.config.num_hidden_layers + 1
    D = model.config.hidden_size
    L.log(f"模型: {model_name}, 层数={n_layers}, 维度={D}")
    
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
    
    all_H_dict = {
        "color": all_H_color,
        "taste": all_H_taste,
        "size": all_H_size,
    }
    
    # 采集名词隐藏状态(用于f_base, 用中间层)
    mid_layer = n_layers // 2
    L.log(f"采集名词隐藏状态(层{mid_layer})...")
    noun_h_dict = collect_noun_hidden_states(model, tokenizer, device, ALL_NOUNS, PROMPT_TEMPLATES_30, mid_layer)
    
    # ========== P318: 全层扫描 ==========
    # 为了节省时间, 只在部分层做扫描(每隔2层)
    L.log("P318: 全层扫描(每2层)...")
    p318_results = {}
    
    test_attrs_short = {
        "color": STIMULI["color_attrs"][:6],
        "taste": STIMULI["taste_attrs"][:6],
        "size": STIMULI["size_attrs"][:6],
    }
    test_nouns_short = {}
    for fam in FAMILY_NAMES:
        test_nouns_short[fam] = STIMULI[fam][::3][:4]
    
    for layer_idx in range(1, n_layers, 2):  # 每2层扫描
        layer_result = {"layer": layer_idx}
        
        for attr_type in ["color", "taste", "size"]:
            labels = labels_dict[attr_type]
            attr_list = test_attrs_short[attr_type]
            
            H_layer = all_H_dict[attr_type][layer_idx]
            G = H_layer
            G_centered, noun_means = noun_centered_G(G, labels)
            Y = build_indicator(labels, STIMULI[f"{attr_type}_attrs"], attr_type)
            
            pca, cca, P_s, U = compute_signal_subspace(G_centered, Y)
            if pca is None:
                layer_result[attr_type] = {"rate": 0}
                continue
            
            global_signal_ops = {}
            for attr in attr_list:
                op = extract_signal_operator(G_centered, labels, attr, attr_type, pca, P_s)
                if op is not None:
                    global_signal_ops[attr] = op
            
            family_fbase = {}
            for fam in FAMILY_NAMES:
                fam_nouns = STIMULI[fam]
                fam_means = {n: noun_means[n] for n in fam_nouns if n in noun_means}
                if fam_means:
                    family_fbase[fam] = np.mean(list(fam_means.values()), axis=0)
            
            total, success = 0, 0
            for fam in FAMILY_NAMES:
                for test_noun in test_nouns_short[fam]:
                    h_noun_data = noun_means.get(test_noun)
                    if h_noun_data is None:
                        continue
                    f_base_noun = noun_means.get(test_noun, np.zeros(D))
                    
                    for test_attr in attr_list:
                        if test_attr not in global_signal_ops:
                            continue
                        
                        logits_new, logits_diff = intervene_with_operator(
                            model, device, h_noun_data, global_signal_ops[test_attr],
                            layer_idx, f_base=f_base_noun, alpha=1.0, beta=1.0)
                        
                        hit, rank, logit_change = check_intervention_success(
                            logits_new, logits_diff, tokenizer, test_attr)
                        
                        if hit:
                            success += 1
                        total += 1
            
            rate = success / total * 100 if total > 0 else 0
            layer_result[attr_type] = {"success": success, "total": total, "rate": rate}
            L.log(f"  L{layer_idx} {attr_type}: {success}/{total} = {rate:.1f}%")
        
        p318_results[layer_idx] = layer_result
    
    # 找最优层(按大小属性, 因为大小干预最成功)
    best_layer = max(p318_results.keys(), 
                     key=lambda l: p318_results[l].get("size", {}).get("rate", 0))
    best_size_rate = p318_results[best_layer].get("size", {}).get("rate", 0)
    L.log(f"最优层: L{best_layer} (大小={best_size_rate:.1f}%)")
    
    # ========== P319: 信号强度缩放 ==========
    L.log("P319: 信号强度缩放...")
    p319_results = {}
    
    alphas = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    betas = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    for attr_type in ["color", "taste", "size"]:
        labels = labels_dict[attr_type]
        attr_list = test_attrs_short[attr_type]
        
        H_layer = all_H_dict[attr_type][best_layer]
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
        
        family_fbase = {}
        for fam in FAMILY_NAMES:
            fam_nouns = STIMULI[fam]
            fam_means = {n: noun_means[n] for n in fam_nouns if n in noun_means}
            if fam_means:
                family_fbase[fam] = np.mean(list(fam_means.values()), axis=0)
        
        attr_results = {}
        for alpha in alphas:
            for beta in betas:
                total, success = 0, 0
                for fam in FAMILY_NAMES:
                    for test_noun in test_nouns_short[fam]:
                        h_noun_data = noun_means.get(test_noun)
                        if h_noun_data is None:
                            continue
                        f_base_noun = noun_means.get(test_noun, np.zeros(D))
                        
                        for test_attr in attr_list:
                            if test_attr not in global_signal_ops:
                                continue
                            
                            logits_new, logits_diff = intervene_with_operator(
                                model, device, h_noun_data, global_signal_ops[test_attr],
                                best_layer, f_base=f_base_noun, alpha=alpha, beta=beta)
                            
                            hit, rank, logit_change = check_intervention_success(
                                logits_new, logits_diff, tokenizer, test_attr)
                            
                            if hit:
                                success += 1
                            total += 1
                
                rate = success / total * 100 if total > 0 else 0
                key = f"a{alpha}_b{beta}"
                attr_results[key] = {"alpha": alpha, "beta": beta, "rate": rate}
        
        # 找最优α,β
        best_key = max(attr_results, key=lambda k: attr_results[k]["rate"])
        best_ab = attr_results[best_key]
        L.log(f"  {attr_type} 最优: α={best_ab['alpha']}, β={best_ab['beta']}, rate={best_ab['rate']:.1f}%")
        p319_results[attr_type] = attr_results
    
    # 确定全局最优α,β(按大小属性)
    size_results = p319_results.get("size", {})
    if size_results:
        best_key = max(size_results, key=lambda k: size_results[k]["rate"])
        best_alpha = size_results[best_key]["alpha"]
        best_beta = size_results[best_key]["beta"]
    else:
        best_alpha, best_beta = 1.0, 1.0
    L.log(f"全局最优: α={best_alpha}, β={best_beta}")
    
    # ========== P320: 多属性联合干预 ==========
    L.log("P320: 多属性联合干预...")
    p320_results = {}
    
    attr_pairs = [
        ("color", "size"),
        ("size", "taste"),
        ("color", "taste"),
    ]
    
    # 预计算每类属性在最优层的信号算子
    signal_ops_by_type = {}
    noun_means_by_type = {}
    for attr_type in ["color", "taste", "size"]:
        labels = labels_dict[attr_type]
        attr_list = STIMULI[f"{attr_type}_attrs"][:4]  # 每类4个属性
        
        H_layer = all_H_dict[attr_type][best_layer]
        G = H_layer
        G_centered, noun_means = noun_centered_G(G, labels)
        Y = build_indicator(labels, STIMULI[f"{attr_type}_attrs"], attr_type)
        
        pca, cca, P_s, U = compute_signal_subspace(G_centered, Y)
        if pca is None:
            continue
        
        ops = {}
        for attr in attr_list:
            op = extract_signal_operator(G_centered, labels, attr, attr_type, pca, P_s)
            if op is not None:
                ops[attr] = op
        
        signal_ops_by_type[attr_type] = ops
        noun_means_by_type[attr_type] = noun_means
    
    for pair in attr_pairs:
        pair_name = "+".join(pair)
        
        total, success_any, success_all = 0, 0, 0
        
        for fam in FAMILY_NAMES:
            for test_noun in test_nouns_short[fam]:
                h_noun_data = noun_means_by_type.get(pair[0], {}).get(test_noun)
                if h_noun_data is None:
                    continue
                f_base_noun = noun_means_by_type.get(pair[0], {}).get(test_noun, np.zeros(D))
                
                # 每个属性取前2个
                from itertools import product
                attrs_per_type = []
                for attr_type in pair:
                    if attr_type in signal_ops_by_type:
                        attrs_per_type.append(list(signal_ops_by_type[attr_type].keys())[:2])
                    else:
                        attrs_per_type.append([])
                
                for combo in product(*attrs_per_type):
                    if any(len(a) == 0 for a in [combo]):
                        continue
                    
                    # 联合算子
                    combined_op = np.zeros(D)
                    target_attrs = []
                    for i, attr_type in enumerate(pair):
                        attr = combo[i]
                        if attr in signal_ops_by_type[attr_type]:
                            combined_op += signal_ops_by_type[attr_type][attr]
                            target_attrs.append(attr)
                    
                    if len(target_attrs) < len(pair):
                        continue
                    
                    logits_new, logits_diff = intervene_with_operator(
                        model, device, h_noun_data, combined_op,
                        best_layer, f_base=f_base_noun, alpha=best_alpha, beta=best_beta)
                    
                    hits = []
                    for attr in target_attrs:
                        hit, _, _ = check_intervention_success(logits_new, logits_diff, tokenizer, attr)
                        hits.append(hit)
                    
                    if any(hits):
                        success_any += 1
                    if all(hits):
                        success_all += 1
                    total += 1
        
        rate_any = success_any / total * 100 if total > 0 else 0
        rate_all = success_all / total * 100 if total > 0 else 0
        p320_results[pair_name] = {
            "rate_any": rate_any, "rate_all": rate_all,
            "success_any": success_any, "success_all": success_all, "total": total
        }
        L.log(f"  {pair_name}: any={rate_any:.1f}%, all={rate_all:.1f}%")
    
    # ========== P321: 大规模跨族泛化 ==========
    L.log("P321: 大规模跨族泛化...")
    p321_results = {}
    
    for attr_type in ["color", "taste", "size"]:
        labels = labels_dict[attr_type]
        attr_list = test_attrs_short[attr_type]
        
        H_layer = all_H_dict[attr_type][best_layer]
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
        
        family_fbase = {}
        for fam in FAMILY_NAMES:
            fam_nouns = STIMULI[fam]
            fam_means = {n: noun_means[n] for n in fam_nouns if n in noun_means}
            if fam_means:
                family_fbase[fam] = np.mean(list(fam_means.values()), axis=0)
        
        # 5×5矩阵(只用fbase+global方法)
        matrix = {}
        for target_fam in FAMILY_NAMES:
            total, success = 0, 0
            
            for test_noun in STIMULI[target_fam]:
                h_noun_data = noun_means.get(test_noun)
                if h_noun_data is None:
                    continue
                
                f_base_target = family_fbase.get(target_fam, np.zeros(D))
                
                for test_attr in attr_list:
                    if test_attr not in global_signal_ops:
                        continue
                    
                    logits_new, logits_diff = intervene_with_operator(
                        model, device, h_noun_data, global_signal_ops[test_attr],
                        best_layer, f_base=f_base_target, alpha=best_alpha, beta=best_beta)
                    
                    hit, rank, logit_change = check_intervention_success(
                        logits_new, logits_diff, tokenizer, test_attr)
                    
                    if hit:
                        success += 1
                    total += 1
            
            rate = success / total * 100 if total > 0 else 0
            matrix[target_fam] = {"success": success, "total": total, "rate": rate}
            L.log(f"  {attr_type} {target_fam[:6]}: {success}/{total} = {rate:.1f}%")
        
        p321_results[attr_type] = matrix
    
    # ========== 保存结果 ==========
    all_results = {
        "model": model_name,
        "n_layers": n_layers,
        "D": D,
        "best_layer": best_layer,
        "best_alpha": best_alpha,
        "best_beta": best_beta,
        "p318_layer_scan": {str(k): v for k, v in p318_results.items()},
        "p319_scaling": p319_results,
        "p320_joint": p320_results,
        "p321_cross_family": p321_results,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M"),
    }
    
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = OUT_DIR / f"phase_lvi_p318_321_{model_name}_{ts}.json"
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
