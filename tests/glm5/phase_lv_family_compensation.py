"""
Phase LV-P315/316/317: 族偏移补偿与跨族泛化
======================================================================

Phase LIV核心结论:
  1. 族内R²比全局高34-126%: 族条件化显著提高编码质量
  2. f_base几乎正交(cos≈0): 族间偏移完全独立
  3. CCA空间通用基底占73-86%: 算子大部分是通用的!
  4. 族特异偏移仅14-27%: 只需补偿偏移即可跨族
  5. 颜色最特异(delta=25-40%), 味道最通用(delta=14-21%)
  6. P311/P314干预bug: 未使用信号子空间投影

Phase LV核心目标: 验证族偏移补偿能否实现跨族泛化

核心假设:
  - 通用算子(73-86%) + 目标族f_base + 目标族Δφ → 跨族干预应该成功
  - h_new = h_noun_target + f_base^(target) + Σ α_i·(φ_i^(global) + Δφ_i^(target))
  - 简化版: h_new = h_noun_target + G_signal^(target)(attr)

三大实验:
  P315: 信号子空间投影的族内干预 — 修复P311的bug
    - 对每个族独立做CCA → 信号子空间投影 → 提取纯净算子
    - h_new = h_noun + G_signal^(k)(attr)
    - 目标: 族内干预成功率>90%

  P316: 族偏移补偿跨族干预 — ★★★核心实验★★★
    - 方法1(直接族算子): h_new = h_noun_target + G_centered^(target)(attr)
      - 只用目标族的算子, 不补偿
    - 方法2(通用+族偏移): h_new = h_noun_target + global_ops(attr) + Δφ^(target)(attr)
      - 用通用算子+目标族偏移
    - 方法3(源族信号): h_new = h_noun_target + G_signal^(source)(attr)
      - 用源族的信号算子(不做补偿)
    - 方法4(目标族信号): h_new = h_noun_target + G_signal^(target)(attr)
      - 用目标族的信号算子(理想情况)
    - 构建5×5×4的干预矩阵

  P317: 大规模干预测试 — 5族×3属性×所有名词×4方法
    - 更大的名词覆盖
    - 5折交叉验证干预成功率
    - 分析失败案例

数据规模: 2160三元组 × 30模板 + 大规模干预
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

LOG_FILE = OUT_DIR / "phase_lv_log.txt"

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
SIZE_LABELS = [(n, "size", s, NOUN_TO_FAMILY.get(s,"unknown")) for n in ALL_NOUNS for s in STIMULI["size_attrs"]]

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
    
    # PCA降维
    pca_dim = min(30, N - 1, D)
    pca = PCA(n_components=pca_dim)
    G_pca = pca.fit_transform(G_centered)
    
    if np.any(np.isnan(G_pca)):
        return None, None, None, None
    
    # CCA
    n_cca = min(n_cca, K - 1, G_pca.shape[1] - 1, N - 1)
    if n_cca < 2:
        return None, None, None, None
    
    try:
        cca = CCA(n_components=n_cca, max_iter=500)
        cca.fit(G_pca, Y)
    except:
        return None, None, None, None
    
    # 信号子空间投影矩阵(在PCA空间)
    U = cca.x_weights_  # (pca_dim, n_cca)
    P_s_pca = U @ np.linalg.inv(U.T @ U + 1e-6 * np.eye(n_cca)) @ U.T  # (pca_dim, pca_dim)
    
    # 在原始空间构造信号投影: P_s = pca.components_.T @ P_s_pca @ pca.components_
    # 但更简单: G_signal = pca.inverse_transform(G_pca @ P_s_pca)
    
    return pca, cca, P_s_pca, U

def extract_signal_operator(G_centered, labels, attr, attr_type, pca, P_s_pca):
    """提取纯净属性算子(在G_centered空间)"""
    attr_mask = [i for i, l in enumerate(labels) if l[2] == attr]
    if len(attr_mask) < 2:
        return None
    
    G_attr = G_centered[attr_mask]
    
    # PCA投影 → 信号子空间投影 → 反投影
    G_pca = pca.transform(G_attr)
    G_signal_pca = G_pca @ P_s_pca
    G_signal = pca.inverse_transform(G_signal_pca)
    
    # 纯净算子 = 信号均值
    operator = np.mean(G_signal, axis=0)
    return operator

def intervene_with_operator(model, tokenizer, device, h_noun, operator, layer, f_base=None):
    """用算子干预, 返回logits_diff(干预-基线)和绝对logits, 含norm层处理"""
    if f_base is not None:
        h_new = h_noun + f_base + operator
    else:
        h_new = h_noun + operator
    
    with torch.no_grad():
        h_tensor = torch.tensor(h_new, dtype=torch.bfloat16).unsqueeze(0).unsqueeze(0).to(device)
        h_base_tensor = torch.tensor(h_noun, dtype=torch.bfloat16).unsqueeze(0).unsqueeze(0).to(device)
        
        lm_head = model.lm_head if hasattr(model, 'lm_head') else model.get_output_embeddings()
        
        # 检查是否有norm层
        norm = None
        if hasattr(model, 'model') and hasattr(model.model, 'norm'):
            norm = model.model.norm
        
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
    """检查干预是否成功: 目标属性词的logit_diff是否在top-k"""
    attr_tok_ids = tokenizer.encode(target_attr, add_special_tokens=False)
    if len(attr_tok_ids) == 0:
        return False, -1, 0.0
    
    attr_tok_id = attr_tok_ids[0]
    
    # 方法1: 检查logits_diff的top-k（Phase LIII的方法）
    top_k_ids_diff = np.argsort(logits_diff)[-top_k:][::-1]
    hit_diff = attr_tok_id in top_k_ids_diff
    rank_diff = int(np.where(top_k_ids_diff == attr_tok_id)[0][0]) + 1 if hit_diff else -1
    
    # 方法2: 检查绝对logits的top-k
    top_k_ids_abs = np.argsort(logits_new)[-top_k:][::-1]
    hit_abs = attr_tok_id in top_k_ids_abs
    
    # logit变化值
    target_logit_change = float(logits_diff[attr_tok_id]) if attr_tok_id < len(logits_diff) else 0.0
    
    return hit_diff, rank_diff, target_logit_change

# ===================== P315: 信号子空间族内干预 =====================
def run_p315_signal_intervention(model, tokenizer, device, G_dict, labels_dict,
                                  attr_type, attr_list, families, layers, L):
    """P315: 用信号子空间投影的族内干预"""
    L.log(f"\n=== P315: 信号子空间族内干预 ({attr_type}) ===")
    
    results = {}
    
    for layer in layers:
        G = G_dict[layer]
        labels = labels_dict[layer] if isinstance(labels_dict, dict) else labels_dict
        N, D = G.shape
        
        if N < 30:
            continue
        
        G_centered, noun_means = noun_centered_G(G, labels)
        if np.any(np.isnan(G_centered)) or np.all(G_centered == 0):
            continue
        
        Y = build_indicator(labels, attr_list, attr_type)
        
        # 全局信号子空间
        pca_g, cca_g, P_s_g, U_g = compute_signal_subspace(G_centered, Y, n_cca=10)
        
        layer_results = {"global": {}, "family": {}}
        
        # 全局算子干预
        global_operators = {}
        for attr in attr_list[:6]:
            op = extract_signal_operator(G_centered, labels, attr, attr_type, pca_g, P_s_g)
            if op is not None:
                global_operators[attr] = op
        
        # 族特异信号子空间和f_base
        family_signal_ops = {}
        family_fbase = {}
        for fam in families:
            fam_mask = [i for i, l in enumerate(labels) if l[3] == fam]
            if len(fam_mask) < 12:
                continue
            
            G_fam = G_centered[fam_mask]
            labels_fam = [labels[i] for i in fam_mask]
            Y_fam = build_indicator(labels_fam, attr_list, attr_type)
            
            pca_f, cca_f, P_s_f, U_f = compute_signal_subspace(G_fam, Y_fam, n_cca=min(10, len(attr_list)-1))
            
            if pca_f is None:
                continue
            
            fam_ops = {}
            for attr in attr_list[:6]:
                op = extract_signal_operator(G_fam, labels_fam, attr, attr_type, pca_f, P_s_f)
                if op is not None:
                    fam_ops[attr] = op
            
            family_signal_ops[fam] = fam_ops
            # f_base: 该族名词的平均偏移(从noun_means提取)
            fam_nouns = set(l[0] for l in labels_fam)
            fam_noun_means = {n: noun_means[n] for n in fam_nouns if n in noun_means}
            if fam_noun_means:
                family_fbase[fam] = np.mean(list(fam_noun_means.values()), axis=0)
        
        # 干预测试: 每个族选4个测试名词
        for fam in families:
            fam_nouns = STIMULI[fam]
            test_nouns = fam_nouns[::3][:4]
            test_attrs = attr_list[:6]
            
            global_success = 0
            family_success = 0
            total = 0
            
            for test_noun in test_nouns:
                try:
                    noun_phrase = f"a {test_noun}"
                    toks = tokenizer(noun_phrase, return_tensors="pt").to(device)
                    with torch.no_grad():
                        out = model(toks.input_ids, output_hidden_states=True)
                        h_noun = out.hidden_states[layer][0, -1].float().cpu().numpy()
                except:
                    continue
                
                # 获取该名词的f_base
                f_base_noun = noun_means.get(test_noun, None)
                if f_base_noun is None:
                    f_base_noun = family_fbase.get(fam, np.zeros_like(h_noun))
                
                for test_attr in test_attrs:
                    # 方法1: 全局信号算子 + f_base
                    if test_attr in global_operators:
                        logits_new, logits_diff = intervene_with_operator(
                            model, tokenizer, device, h_noun, global_operators[test_attr], layer, f_base=f_base_noun)
                        hit, rank, logit_change = check_intervention_success(
                            logits_new, logits_diff, tokenizer, test_attr)
                        if hit:
                            global_success += 1
                    
                    # 方法2: 族特异信号算子 + f_base
                    if fam in family_signal_ops and test_attr in family_signal_ops[fam]:
                        logits_new, logits_diff = intervene_with_operator(
                            model, tokenizer, device, h_noun, family_signal_ops[fam][test_attr], layer, f_base=f_base_noun)
                        hit, rank, logit_change = check_intervention_success(
                            logits_new, logits_diff, tokenizer, test_attr)
                        if hit:
                            family_success += 1
                    
                    total += 1
            
            if total > 0:
                layer_results["global"][fam] = {
                    "success": global_success,
                    "total": total,
                    "rate": global_success / total
                }
                layer_results["family"][fam] = {
                    "success": family_success,
                    "total": total,
                    "rate": family_success / total
                }
                L.log(f"  L{layer} {fam}: global={global_success}/{total}({global_success/total:.1%}), "
                      f"family={family_success}/{total}({family_success/total:.1%})")
        
        results[layer] = layer_results
    
    return results

# ===================== P316: 族偏移补偿跨族干预 =====================
def run_p316_cross_family_compensation(model, tokenizer, device, G_dict, labels_dict,
                                        attr_type, attr_list, families, layers, L):
    """P316: 4种跨族干预方法比较"""
    L.log(f"\n=== P316: 族偏移补偿跨族干预 ({attr_type}) ===")
    
    results = {}
    
    for layer in layers:
        G = G_dict[layer]
        labels = labels_dict[layer] if isinstance(labels_dict, dict) else labels_dict
        N, D = G.shape
        
        if N < 30:
            continue
        
        G_centered, noun_means = noun_centered_G(G, labels)
        if np.any(np.isnan(G_centered)) or np.all(G_centered == 0):
            continue
        
        Y = build_indicator(labels, attr_list, attr_type)
        
        # 全局信号子空间和算子
        pca_g, cca_g, P_s_g, U_g = compute_signal_subspace(G_centered, Y, n_cca=10)
        if pca_g is None:
            continue
        
        global_signal_ops = {}
        global_centered_ops = {}  # 不投影的原始均值
        for attr in attr_list[:6]:
            op_sig = extract_signal_operator(G_centered, labels, attr, attr_type, pca_g, P_s_g)
            if op_sig is not None:
                global_signal_ops[attr] = op_sig
            # 原始均值
            attr_mask = [i for i, l in enumerate(labels) if l[2] == attr]
            if len(attr_mask) >= 2:
                global_centered_ops[attr] = np.mean(G_centered[attr_mask], axis=0)
        
        # 族特异信号子空间和算子
        family_data = {}
        for fam in families:
            fam_mask = [i for i, l in enumerate(labels) if l[3] == fam]
            if len(fam_mask) < 12:
                continue
            
            G_fam = G_centered[fam_mask]
            labels_fam = [labels[i] for i in fam_mask]
            Y_fam = build_indicator(labels_fam, attr_list, attr_type)
            
            pca_f, cca_f, P_s_f, U_f = compute_signal_subspace(G_fam, Y_fam, n_cca=min(10, len(attr_list)-1))
            
            fam_signal_ops = {}
            fam_centered_ops = {}
            for attr in attr_list[:6]:
                if pca_f is not None:
                    op = extract_signal_operator(G_fam, labels_fam, attr, attr_type, pca_f, P_s_f)
                    if op is not None:
                        fam_signal_ops[attr] = op
                # 族内原始均值
                attr_mask_f = [i for i, l in enumerate(labels_fam) if l[2] == attr]
                if len(attr_mask_f) >= 2:
                    fam_centered_ops[attr] = np.mean(G_fam[attr_mask_f], axis=0)
            
            # 族偏移: Δφ^(k) = fam_ops - global_ops
            delta_ops = {}
            for attr in attr_list[:6]:
                if attr in fam_centered_ops and attr in global_centered_ops:
                    delta_ops[attr] = fam_centered_ops[attr] - global_centered_ops[attr]
            
            # f_base: 族内名词的平均偏移
            fam_nouns_set = set(l[0] for l in labels_fam)
            fam_noun_means = {n: noun_means[n] for n in fam_nouns_set if n in noun_means}
            fbase = np.mean(list(fam_noun_means.values()), axis=0) if fam_noun_means else np.zeros(D)
            
            family_data[fam] = {
                "signal_ops": fam_signal_ops,
                "centered_ops": fam_centered_ops,
                "delta_ops": delta_ops,
                "fbase": fbase
            }
        
        # 跨族干预测试: 5×5矩阵
        transfer_results = {}
        source_fams = sorted(family_data.keys())
        target_fams = sorted(families)
        test_attrs = attr_list[:6]
        
        for source_fam in source_fams:
            transfer_results[source_fam] = {}
            
            for target_fam in target_fams:
                target_nouns = STIMULI[target_fam]
                # 选3个测试名词(跳过前3个, 避免训练集重叠)
                test_nouns = target_nouns[3:6] if len(target_nouns) >= 6 else target_nouns[:3]
                
                method_results = {
                    "M1_source_signal": 0,  # 源族信号算子(无补偿)
                    "M2_global_signal": 0,  # 全局信号算子
                    "M3_global_plus_delta": 0,  # 全局+目标族偏移
                    "M4_target_signal": 0,  # 目标族信号算子(上界)
                }
                total = 0
                
                for test_noun in test_nouns:
                    try:
                        noun_phrase = f"a {test_noun}"
                        toks = tokenizer(noun_phrase, return_tensors="pt").to(device)
                        with torch.no_grad():
                            out = model(toks.input_ids, output_hidden_states=True)
                            h_noun = out.hidden_states[layer][0, -1].float().cpu().numpy()
                    except:
                        continue
                    
                for test_attr in test_attrs:
                    # 获取目标名词的f_base
                    f_base_noun = noun_means.get(test_noun, None)
                    if f_base_noun is None:
                        f_base_noun = family_data.get(target_fam, {}).get("fbase", np.zeros(D))
                    
                    # M1: 源族信号算子(不做任何补偿) + f_base
                    if test_attr in family_data[source_fam].get("signal_ops", {}):
                        logits_new, logits_diff = intervene_with_operator(model, tokenizer, device,
                            h_noun, family_data[source_fam]["signal_ops"][test_attr], layer, f_base=f_base_noun)
                        hit, _, _ = check_intervention_success(logits_new, logits_diff, tokenizer, test_attr)
                        if hit:
                            method_results["M1_source_signal"] += 1
                    
                    # M2: 全局信号算子 + f_base
                    if test_attr in global_signal_ops:
                        logits_new, logits_diff = intervene_with_operator(model, tokenizer, device,
                            h_noun, global_signal_ops[test_attr], layer, f_base=f_base_noun)
                        hit, _, _ = check_intervention_success(logits_new, logits_diff, tokenizer, test_attr)
                        if hit:
                            method_results["M2_global_signal"] += 1
                    
                    # M3: 全局+目标族偏移 + f_base(核心方法!)
                    if (test_attr in global_centered_ops and 
                        target_fam in family_data and
                        test_attr in family_data[target_fam].get("delta_ops", {})):
                        op_compensated = global_centered_ops[test_attr] + family_data[target_fam]["delta_ops"][test_attr]
                        logits_new, logits_diff = intervene_with_operator(model, tokenizer, device,
                            h_noun, op_compensated, layer, f_base=f_base_noun)
                        hit, _, _ = check_intervention_success(logits_new, logits_diff, tokenizer, test_attr)
                        if hit:
                            method_results["M3_global_plus_delta"] += 1
                    
                    # M4: 目标族信号算子 + f_base(上界)
                    if (target_fam in family_data and 
                        test_attr in family_data[target_fam].get("signal_ops", {})):
                        logits_new, logits_diff = intervene_with_operator(model, tokenizer, device,
                            h_noun, family_data[target_fam]["signal_ops"][test_attr], layer, f_base=f_base_noun)
                        hit, _, _ = check_intervention_success(logits_new, logits_diff, tokenizer, test_attr)
                        if hit:
                            method_results["M4_target_signal"] += 1
                        
                        total += 1
                
                if total > 0:
                    transfer_results[source_fam][target_fam] = {
                        "total": total,
                        "methods": {k: {"success": v, "rate": v/total} for k, v in method_results.items()}
                    }
        
        results[layer] = transfer_results
        
        # 打印跨族干预矩阵
        L.log(f"\n  L{layer} Cross-family intervention matrix ({attr_type}):")
        
        for method_name, method_label in [("M1_source_signal", "M1:源族信号"), 
                                           ("M2_global_signal", "M2:全局信号"),
                                           ("M3_global_plus_delta", "M3:全局+Δ"),
                                           ("M4_target_signal", "M4:目标族信号")]:
            L.log(f"  {method_label}:")
            header = "  Source\\Target\t" + "\t".join([f.replace("_family","")[:4] for f in target_fams])
            L.log(header)
            for sf in source_fams:
                row = f"  {sf.replace('_family','')[:4]}\t\t"
                for tf in target_fams:
                    if sf in transfer_results and tf in transfer_results[sf]:
                        rate = transfer_results[sf][tf]["methods"][method_name]["rate"]
                        row += f"{rate:.0%}\t\t"
                    else:
                        row += "N/A\t\t"
                L.log(row)
    
    return results

# ===================== P317: 大规模干预测试 =====================
def run_p317_large_scale_intervention(model, tokenizer, device, G_dict, labels_dict,
                                       attr_type, attr_list, families, layers, L):
    """P317: 更大规模的干预测试, 含详细分析"""
    L.log(f"\n=== P317: 大规模干预测试 ({attr_type}) ===")
    
    results = {}
    
    for layer in layers:
        G = G_dict[layer]
        labels = labels_dict[layer] if isinstance(labels_dict, dict) else labels_dict
        N, D = G.shape
        
        if N < 30:
            continue
        
        G_centered, noun_means = noun_centered_G(G, labels)
        if np.any(np.isnan(G_centered)) or np.all(G_centered == 0):
            continue
        
        Y = build_indicator(labels, attr_list, attr_type)
        
        # 全局信号子空间
        pca_g, cca_g, P_s_g, U_g = compute_signal_subspace(G_centered, Y, n_cca=10)
        if pca_g is None:
            continue
        
        # 族特异数据
        family_data = {}
        for fam in families:
            fam_mask = [i for i, l in enumerate(labels) if l[3] == fam]
            if len(fam_mask) < 12:
                continue
            
            G_fam = G_centered[fam_mask]
            labels_fam = [labels[i] for i in fam_mask]
            Y_fam = build_indicator(labels_fam, attr_list, attr_type)
            
            pca_f, cca_f, P_s_f, U_f = compute_signal_subspace(G_fam, Y_fam, n_cca=min(10, len(attr_list)-1))
            
            fam_signal_ops = {}
            fam_centered_ops = {}
            for attr in attr_list[:6]:
                if pca_f is not None:
                    op = extract_signal_operator(G_fam, labels_fam, attr, attr_type, pca_f, P_s_f)
                    if op is not None:
                        fam_signal_ops[attr] = op
                attr_mask_f = [i for i, l in enumerate(labels_fam) if l[2] == attr]
                if len(attr_mask_f) >= 2:
                    fam_centered_ops[attr] = np.mean(G_fam[attr_mask_f], axis=0)
            
            delta_ops = {}
            for attr in attr_list[:6]:
                attr_mask_g = [i for i, l in enumerate(labels) if l[2] == attr]
                if len(attr_mask_g) >= 2 and attr in fam_centered_ops:
                    global_mean = np.mean(G_centered[attr_mask_g], axis=0)
                    delta_ops[attr] = fam_centered_ops[attr] - global_mean
            
            family_data[fam] = {
                "signal_ops": fam_signal_ops,
                "centered_ops": fam_centered_ops,
                "delta_ops": delta_ops,
                "fbase": np.mean(G_fam, axis=0)  # 族的平均偏移
            }
        
        # 大规模干预: 所有族的所有名词
        all_intervention_results = []
        
        for target_fam in families:
            target_nouns = STIMULI[target_fam]
            test_attrs = attr_list[:6]
            
            for test_noun in target_nouns:
                try:
                    noun_phrase = f"a {test_noun}"
                    toks = tokenizer(noun_phrase, return_tensors="pt").to(device)
                    with torch.no_grad():
                        out = model(toks.input_ids, output_hidden_states=True)
                        h_noun = out.hidden_states[layer][0, -1].float().cpu().numpy()
                except:
                    continue
                
                for test_attr in test_attrs:
                    result_entry = {
                        "noun": test_noun,
                        "attr": test_attr,
                        "target_family": target_fam,
                        "methods": {}
                    }
                    
                    # 获取f_base
                    f_base_noun = noun_means.get(test_noun, None)
                    if f_base_noun is None:
                        f_base_noun = family_data.get(target_fam, {}).get("fbase", np.zeros(D))
                    
                    # 方法1: 目标族信号算子 + f_base(最佳上界)
                    if target_fam in family_data and test_attr in family_data[target_fam].get("signal_ops", {}):
                        logits_new, logits_diff = intervene_with_operator(model, tokenizer, device,
                            h_noun, family_data[target_fam]["signal_ops"][test_attr], layer, f_base=f_base_noun)
                        hit, rank, logit_change = check_intervention_success(logits_new, logits_diff, tokenizer, test_attr)
                        result_entry["methods"]["target_signal"] = {"hit": hit, "rank": rank, "logit_change": logit_change}
                    
                    # 方法2: 全局+目标族偏移 + f_base
                    if target_fam in family_data and test_attr in family_data[target_fam].get("delta_ops", {}):
                        attr_mask_g = [i for i, l in enumerate(labels) if l[2] == test_attr]
                        if len(attr_mask_g) >= 2:
                            global_mean = np.mean(G_centered[attr_mask_g], axis=0)
                            op_comp = global_mean + family_data[target_fam]["delta_ops"][test_attr]
                            logits_new, logits_diff = intervene_with_operator(model, tokenizer, device,
                                h_noun, op_comp, layer, f_base=f_base_noun)
                            hit, rank, logit_change = check_intervention_success(logits_new, logits_diff, tokenizer, test_attr)
                            result_entry["methods"]["global_plus_delta"] = {"hit": hit, "rank": rank, "logit_change": logit_change}
                    
                    # 方法3: 全局信号算子 + f_base
                    op_sig = extract_signal_operator(G_centered, labels, test_attr, attr_type, pca_g, P_s_g)
                    if op_sig is not None:
                        logits_new, logits_diff = intervene_with_operator(model, tokenizer, device,
                            h_noun, op_sig, layer, f_base=f_base_noun)
                        hit, rank, logit_change = check_intervention_success(logits_new, logits_diff, tokenizer, test_attr)
                        result_entry["methods"]["global_signal"] = {"hit": hit, "rank": rank, "logit_change": logit_change}
                    
                    # 方法4: 目标族f_base + 全局信号(含f_base叠加)
                    if target_fam in family_data:
                        op_with_fbase = family_data[target_fam]["fbase"] + (op_sig if op_sig is not None else 0)
                        logits_new, logits_diff = intervene_with_operator(model, tokenizer, device,
                            h_noun, op_with_fbase, layer)
                        hit, rank, logit_change = check_intervention_success(logits_new, logits_diff, tokenizer, test_attr)
                        result_entry["methods"]["fbase_plus_global_signal"] = {"hit": hit, "rank": rank, "logit_change": logit_change}
                    
                    all_intervention_results.append(result_entry)
        
        # 汇总统计
        method_stats = {}
        for method in ["target_signal", "global_plus_delta", "global_signal", "fbase_plus_global_signal"]:
            hits = sum(1 for r in all_intervention_results 
                      if method in r["methods"] and r["methods"][method]["hit"])
            total = sum(1 for r in all_intervention_results if method in r["methods"])
            method_stats[method] = {
                "success": hits,
                "total": total,
                "rate": hits / total if total > 0 else 0.0
            }
        
        # 按族统计
        family_stats = {}
        for fam in families:
            fam_results = [r for r in all_intervention_results if r["target_family"] == fam]
            for method in ["target_signal", "global_plus_delta", "global_signal"]:
                hits = sum(1 for r in fam_results if method in r["methods"] and r["methods"][method]["hit"])
                total = sum(1 for r in fam_results if method in r["methods"])
                if fam not in family_stats:
                    family_stats[fam] = {}
                family_stats[fam][method] = {
                    "success": hits,
                    "total": total,
                    "rate": hits / total if total > 0 else 0.0
                }
        
        results[layer] = {
            "method_stats": method_stats,
            "family_stats": family_stats,
            "n_interventions": len(all_intervention_results)
        }
        
        L.log(f"  L{layer} Method stats:")
        for method, stats in method_stats.items():
            L.log(f"    {method}: {stats['success']}/{stats['total']} ({stats['rate']:.1%})")
        
        L.log(f"  L{layer} Family stats (target_signal):")
        for fam, stats in family_stats.items():
            ts = stats.get("target_signal", {})
            gd = stats.get("global_plus_delta", {})
            L.log(f"    {fam}: target_signal={ts.get('rate',0):.1%}, global_plus_delta={gd.get('rate',0):.1%}")
    
    return results

# ===================== 主函数 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3", choices=["qwen3","glm4","deepseek7b"])
    args = parser.parse_args()
    
    model_name = args.model
    L.log(f"=== Phase LV: 族偏移补偿与跨族泛化 ({model_name}) ===")
    L.log(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 加载模型
    L.log("Loading model...")
    model, tokenizer, device = load_model(model_name)
    n_layers = len(model.model.layers) if hasattr(model, 'model') and hasattr(model.model, 'layers') else 0
    L.log(f"Model loaded: {n_layers} layers, device={device}")
    
    # 选择测试层(只测末2层, 加速)
    if n_layers > 0:
        last_layer = n_layers - 1
        test_layers = [max(0, last_layer - 3), last_layer]
    else:
        test_layers = [28, 32]
    
    L.log(f"Test layers: {test_layers}")
    
    # 构建三元组
    all_labels = []
    for n in ALL_NOUNS:
        for c in STIMULI["color_attrs"]:
            all_labels.append((n, "color", c, NOUN_TO_FAMILY.get(n, "unknown")))
        for t in STIMULI["taste_attrs"]:
            all_labels.append((n, "taste", t, NOUN_TO_FAMILY.get(n, "unknown")))
        for s in STIMULI["size_attrs"]:
            all_labels.append((n, "size", s, NOUN_TO_FAMILY.get(n, "unknown")))
    
    L.log(f"Total labels: {len(all_labels)}")
    
    all_results = {}
    
    for attr_type, attr_list in [("color", STIMULI["color_attrs"]),
                                  ("taste", STIMULI["taste_attrs"]),
                                  ("size", STIMULI["size_attrs"])]:
        L.log(f"\n{'='*60}")
        L.log(f"Processing {attr_type}...")
        
        attr_labels = [l for l in all_labels if l[1] == attr_type]
        attr_triples = [(l[0], l[2], f"{l[2]} {l[0]}") for l in attr_labels]
        
        # 收集G项
        L.log(f"Collecting {attr_type} G terms...")
        
        G_dict = {}
        labels_dict = {}
        
        for layer in test_layers:
            L.log(f"  Collecting L{layer}...")
            G_all = []
            nouns_only = sorted(set([(t[0], f"a {t[0]}") for t in attr_triples]))
            noun_cache = {}
            
            with torch.no_grad():
                for noun, noun_phrase in nouns_only:
                    toks = tokenizer(noun_phrase, return_tensors="pt").to(device)
                    out = model(toks.input_ids, output_hidden_states=True)
                    noun_cache[noun] = out.hidden_states[layer][0, -1].float().cpu().numpy()
                
                for noun, attr, phrase in attr_triples:
                    toks = tokenizer(phrase, return_tensors="pt").to(device)
                    out = model(toks.input_ids, output_hidden_states=True)
                    h_comb = out.hidden_states[layer][0, -1].float().cpu().numpy()
                    G = h_comb - noun_cache[noun]
                    G_all.append(G)
            
            G_dict[layer] = np.array(G_all)
            labels_dict[layer] = attr_labels
        
        # P315: 信号子空间族内干预
        p315 = run_p315_signal_intervention(model, tokenizer, device, G_dict, labels_dict,
                                             attr_type, attr_list, FAMILY_NAMES, test_layers, L)
        all_results[f"p315_{attr_type}"] = p315
        
        # P316: 族偏移补偿跨族干预
        p316 = run_p316_cross_family_compensation(model, tokenizer, device, G_dict, labels_dict,
                                                    attr_type, attr_list, FAMILY_NAMES, test_layers, L)
        all_results[f"p316_{attr_type}"] = p316
        
        # P317: 大规模干预测试
        p317 = run_p317_large_scale_intervention(model, tokenizer, device, G_dict, labels_dict,
                                                   attr_type, attr_list, FAMILY_NAMES, test_layers, L)
        all_results[f"p317_{attr_type}"] = p317
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    out_file = OUT_DIR / f"phase_lv_p315_317_{model_name}_{timestamp}.json"
    
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        if isinstance(obj, bool):
            return obj
        return obj
    
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(convert(all_results), f, indent=2, ensure_ascii=False)
    
    L.log(f"\nResults saved to {out_file}")
    L.log(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.close()
    
    # 清理
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"\n{'='*60}")
    print(f"Phase LV ({model_name}) completed!")
    print(f"Results: {out_file}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
