"""
Phase LIV-P310/311/312/313/314: 族条件化建模 — 条件化语义物理
======================================================================

Phase LIII核心结论:
  1. 大小算子干预96%成功(GLM4): 白盒算子确实可控
  2. 跨名词族泛化0-17%: 算子是族特异的, 非通用"语义基石"
  3. 噪声是混合结构: 30-50%通用 + 50-70%属性特异
  4. 大小>颜色>味道: 可控性与属性"物理性"正相关
  5. G项 = f_base(noun_family) + Σ α_i·φ_i(attr) + ε

Phase LIV核心目标: 建立族条件化语义物理公式
  h = h_noun + f_base^(k)(noun) + Σ α_i·φ_i^(k)(attr) + ε^(k)

  其中 k 是名词族索引, φ_i^(k) 是第k族的信号算子

核心假设:
  - 每个族有自己的信号算子, 族内应该比全局算子更好
  - 族间算子差异揭示编码策略的异质性
  - 可能存在"通用基底" + "族特异偏移"的混合结构

五大实验:
  P310: 族条件化CCA — 为每个族独立建模信号算子
    - 对5个名词族, 分别做CCA: G^(k)_centered vs Y^(k)
    - 比较族内CCA vs 全局CCA的相关系数和R²
    - 预期: 族内CCA R² > 全局CCA R²

  P311: 族内干预实验 — 验证族特异算子的可控性
    - 用族特异算子 φ_i^(k) 替代全局算子
    - h_new = h_noun + f_base^(k) + G_signal^(k)(attr)
    - 目标: 族内干预成功率>90%

  P312: 族间编码差异分析 — 理解为什么跨族不泛化
    - 比较不同族的CCA方向 u_i^(k) 的余弦相似度
    - 分析f_base^(k)的方向差异
    - 量化族间算子的"旋转角度"

  P313: 通用基底+族特异偏移的混合模型
    - 假设: φ_i^(k) = φ_i^(global) + Δφ_i^(k)
    - 用PCA提取所有族的"通用基底"
    - 分析Δφ_i^(k)的方差贡献比
    - 如果通用基底贡献>80%: 算子本质通用, 只是族偏移
    - 如果通用基底贡献<50%: 算子本质不同, 需要独立建模

  P314: 大规模五族泛化测试
    - 5族×3属性×12名词×3层×3模型 = 1620个干预测试
    - 每个族用本族算子做干预, 测试族内泛化率
    - 交叉族干预: 每个族的算子注入其他4族名词
    - 构建5×5的"族间干预转移矩阵"

数据规模: 2160三元组 × 30模板 = 67650前向传播 + 干预实验
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

LOG_FILE = OUT_DIR / "phase_liv_log.txt"

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

# ===================== 前向传播收集 =====================
def collect_residuals(model, tokenizer, device, templates, triples, last_token_pos=0):
    """收集残差流, 返回G项 = h(attr+noun) - h(noun)"""
    G_all = []
    nouns_only = sorted(set([(t[0], f"a {t[0]}") for t in triples]))
    noun_cache = {}
    
    with torch.no_grad():
        # 先收集纯名词表示
        for noun, noun_phrase in nouns_only:
            toks = tokenizer(noun_phrase, return_tensors="pt").to(device)
            out = model(toks.input_ids, output_hidden_states=True)
            h_noun = out.hidden_states[-1][0, -1].float().cpu().numpy()
            noun_cache[noun] = h_noun
        
        # 收集属性+名词组合
        for noun, attr, phrase in triples:
            toks = tokenizer(phrase, return_tensors="pt").to(device)
            out = model(toks.input_ids, output_hidden_states=True)
            h_comb = out.hidden_states[-1][0, -1].float().cpu().numpy()
            G = h_comb - noun_cache[noun]
            G_all.append(G)
    
    return np.array(G_all)

def collect_residuals_all_layers(model, tokenizer, device, templates, triples, layers):
    """收集所有指定层的残差流"""
    results = {l: [] for l in layers}
    nouns_only = sorted(set([(t[0], f"a {t[0]}") for t in triples]))
    noun_cache = {l: {} for l in layers}
    
    with torch.no_grad():
        # 先收集纯名词表示
        for noun, noun_phrase in nouns_only:
            toks = tokenizer(noun_phrase, return_tensors="pt").to(device)
            out = model(toks.input_ids, output_hidden_states=True)
            for l in layers:
                h_noun = out.hidden_states[l][0, -1].float().cpu().numpy()
                noun_cache[l][noun] = h_noun
        
        # 收集属性+名词组合
        for noun, attr, phrase in triples:
            toks = tokenizer(phrase, return_tensors="pt").to(device)
            out = model(toks.input_ids, output_hidden_states=True)
            for l in layers:
                h_comb = out.hidden_states[l][0, -1].float().cpu().numpy()
                G = h_comb - noun_cache[l][noun]
                results[l].append(G)
    
    return {l: np.array(results[l]) for l in layers}

# ===================== P310: 族条件化CCA =====================
def run_p310_family_cca(G_dict, labels_dict, attr_type, attr_list, families, layers, L):
    """为每个族独立建模CCA, 比较族内vs全局"""
    L.log(f"\n=== P310: 族条件化CCA ({attr_type}) ===")
    
    results = {"global": {}, "family": {}}
    
    for layer in layers:
        G = G_dict[layer]
        labels = labels_dict[layer] if isinstance(labels_dict, dict) else labels_dict
        N, D = G.shape
        
        if N < 30:
            continue
        
        # 去名词均值
        G_centered, noun_means = noun_centered_G(G, labels)
        if np.any(np.isnan(G_centered)) or np.all(G_centered == 0):
            continue
        
        # PCA降维
        pca_dim = min(30, N - 1, D)
        pca = PCA(n_components=pca_dim)
        G_pca = pca.fit_transform(G_centered)
        if np.any(np.isnan(G_pca)) or np.any(np.isinf(G_pca)):
            continue
        
        # 全局CCA
        Y_global = build_indicator(labels, attr_list, attr_type)
        n_cca = min(10, len(attr_list) - 1, G_pca.shape[1] - 1, N - 1)
        try:
            cca_global = CCA(n_components=n_cca, max_iter=500)
            G_cca_g, Y_cca_g = cca_global.fit_transform(G_pca, Y_global)
            corrs_global = [np.corrcoef(G_cca_g[:, i], Y_cca_g[:, i])[0, 1] for i in range(n_cca)]
            # 重构R²
            ridge_global = Ridge(alpha=1.0)
            ridge_global.fit(Y_global, G_pca)
            r2_global = ridge_global.score(Y_global, G_pca)
        except:
            corrs_global = [0.0] * n_cca
            r2_global = 0.0
        
        results["global"][layer] = {
            "cca_corrs": corrs_global,
            "r2": r2_global,
            "n_samples": N
        }
        
        # 族条件化CCA
        for fam in families:
            fam_mask = [i for i, l in enumerate(labels) if l[3] == fam]
            if len(fam_mask) < 15:
                continue
            
            G_fam = G_pca[fam_mask]
            Y_fam = build_indicator([labels[i] for i in fam_mask], attr_list, attr_type)
            
            n_cca_f = min(10, len(attr_list) - 1, G_fam.shape[1] - 1, len(fam_mask) - 1)
            if n_cca_f < 2:
                continue
            
            try:
                cca_fam = CCA(n_components=n_cca_f, max_iter=500)
                G_cca_f, Y_cca_f = cca_fam.fit_transform(G_fam, Y_fam)
                corrs_fam = [np.corrcoef(G_cca_f[:, i], Y_cca_f[:, i])[0, 1] for i in range(n_cca_f)]
                ridge_fam = Ridge(alpha=1.0)
                ridge_fam.fit(Y_fam, G_fam)
                r2_fam = ridge_fam.score(Y_fam, G_fam)
                
                # 5折交叉验证
                cv_scores = []
                kf = KFold(n_splits=min(5, len(fam_mask) // 5), shuffle=True, random_state=42)
                for train_idx, test_idx in kf.split(G_fam):
                    try:
                        ridge_cv = Ridge(alpha=1.0)
                        ridge_cv.fit(Y_fam[train_idx], G_fam[train_idx])
                        cv_scores.append(ridge_cv.score(Y_fam[test_idx], G_fam[test_idx]))
                    except:
                        pass
                cv_r2 = np.mean(cv_scores) if cv_scores else 0.0
                
            except Exception as e:
                corrs_fam = [0.0] * n_cca_f
                r2_fam = 0.0
                cv_r2 = 0.0
            
            if fam not in results["family"]:
                results["family"][fam] = {}
            
            results["family"][fam][layer] = {
                "cca_corrs": corrs_fam,
                "r2": r2_fam,
                "cv_r2": cv_r2,
                "n_samples": len(fam_mask)
            }
            
            L.log(f"  L{layer} {fam}: corrs={[f'{c:.3f}' for c in corrs_fam[:5]]}, r2={r2_fam:.4f}, cv_r2={cv_r2:.4f}, N={len(fam_mask)}")
        
        L.log(f"  L{layer} global: corrs={[f'{c:.3f}' for c in corrs_global[:5]]}, r2={r2_global:.4f}")
    
    return results

# ===================== P311: 族内干预实验 =====================
def run_p311_family_intervention(model, tokenizer, device, G_dict, labels_dict, 
                                  attr_type, attr_list, families, test_layers, L):
    """用族特异算子做族内干预"""
    L.log(f"\n=== P311: 族内干预实验 ({attr_type}) ===")
    
    results = {}
    
    for layer in test_layers:
        G = G_dict[layer]
        labels = labels_dict[layer] if isinstance(labels_dict, dict) else labels_dict
        N, D = G.shape
        
        if N < 30:
            continue
        
        # 全局建模
        G_centered, noun_means = noun_centered_G(G, labels)
        if np.any(np.isnan(G_centered)) or np.all(G_centered == 0):
            continue
        
        pca_dim = min(30, N - 1, D)
        pca = PCA(n_components=pca_dim)
        G_pca = pca.fit_transform(G_centered)
        if np.any(np.isnan(G_pca)):
            continue
        
        Y_all = build_indicator(labels, attr_list, attr_type)
        n_cca = min(10, len(attr_list) - 1, G_pca.shape[1] - 1)
        
        try:
            cca = CCA(n_components=n_cca, max_iter=500)
            cca.fit(G_pca, Y_all)
        except:
            continue
        
        # 全局算子
        global_operators = {}
        for attr in attr_list:
            attr_mask = [i for i, l in enumerate(labels) if l[2] == attr]
            if len(attr_mask) < 3:
                continue
            mean_signal = np.mean(G_centered[attr_mask], axis=0)
            global_operators[attr] = mean_signal
        
        # 族特异算子
        family_operators = {}
        family_pcas = {}
        
        for fam in families:
            fam_mask = [i for i, l in enumerate(labels) if l[3] == fam]
            if len(fam_mask) < 12:
                continue
            
            G_fam = G_centered[fam_mask]
            labels_fam = [labels[i] for i in fam_mask]
            
            # 族内PCA
            pca_dim_f = min(30, len(fam_mask) - 1, D)
            pca_f = PCA(n_components=pca_dim_f)
            G_pca_f = pca_f.fit_transform(G_fam)
            
            if np.any(np.isnan(G_pca_f)):
                continue
            
            # 族内CCA
            Y_fam = build_indicator(labels_fam, attr_list, attr_type)
            n_cca_f = min(10, len(attr_list) - 1, G_pca_f.shape[1] - 1)
            
            fam_ops = {}
            try:
                cca_f = CCA(n_components=n_cca_f, max_iter=500)
                cca_f.fit(G_pca_f, Y_fam)
                
                # 构造信号子空间投影
                U_f = cca_f.x_weights_  # (pca_dim_f, n_cca_f)
                P_s_f = U_f @ np.linalg.inv(U_f.T @ U_f + 1e-6 * np.eye(n_cca_f)) @ U_f.T
                
                for attr in attr_list:
                    attr_mask_f = [i for i, l in enumerate(labels_fam) if l[2] == attr]
                    if len(attr_mask_f) < 2:
                        continue
                    mean_sig_f = np.mean(G_fam[attr_mask_f], axis=0)
                    fam_ops[attr] = mean_sig_f
                    
            except:
                for attr in attr_list:
                    attr_mask_f = [i for i, l in enumerate(labels_fam) if l[2] == attr]
                    if len(attr_mask_f) < 2:
                        continue
                    fam_ops[attr] = np.mean(G_fam[attr_mask_f], axis=0)
            
            family_operators[fam] = fam_ops
            family_pcas[fam] = pca_f
        
        # 干预测试: 每个族选2个测试名词×3个测试属性
        layer_result = {"global": {}, "family": {}}
        
        test_attrs = attr_list[:3]  # 前3个属性
        
        for fam in families:
            fam_nouns = [n for n in STIMULI[fam]]
            if len(fam_nouns) < 4:
                continue
            
            # 选测试名词(后半部分)
            test_nouns = fam_nouns[len(fam_nouns)//2:]
            
            fam_global_success = 0
            fam_family_success = 0
            fam_total = 0
            
            for test_noun in test_nouns[:2]:
                for test_attr in test_attrs:
                    try:
                        # 获取noun基线
                        noun_phrase = f"a {test_noun}"
                        toks = tokenizer(noun_phrase, return_tensors="pt").to(device)
                        with torch.no_grad():
                            out = model(toks.input_ids, output_hidden_states=True)
                            h_noun = out.hidden_states[layer][0, -1].float().cpu().numpy()
                        
                        # 方法1: 全局算子干预
                        if test_attr in global_operators:
                            h_new_global = h_noun + global_operators[test_attr]
                        else:
                            h_new_global = h_noun
                        
                        # 方法2: 族特异算子干预
                        if fam in family_operators and test_attr in family_operators[fam]:
                            h_new_family = h_noun + family_operators[fam][test_attr]
                        else:
                            h_new_family = h_noun
                        
                        # 通过lm_head获取logits
                        with torch.no_grad():
                            h_tensor_g = torch.tensor(h_new_global, dtype=torch.bfloat16).unsqueeze(0).unsqueeze(0).to(device)
                            h_tensor_f = torch.tensor(h_new_family, dtype=torch.bfloat16).unsqueeze(0).unsqueeze(0).to(device)
                            
                            lm_head = model.lm_head if hasattr(model, 'lm_head') else model.get_output_embeddings()
                            logits_g = lm_head(h_tensor_g)[0, 0].float().cpu().numpy()
                            logits_f = lm_head(h_tensor_f)[0, 0].float().cpu().numpy()
                        
                        # 检查目标属性词的logit排名
                        attr_tok_id = tokenizer.encode(f" {test_attr}", add_special_tokens=False)
                        if len(attr_tok_id) > 0:
                            attr_tok_id = attr_tok_id[0]
                        else:
                            continue
                        
                        top_k = 20
                        top_k_ids_g = np.argsort(logits_g)[-top_k:][::-1]
                        top_k_ids_f = np.argsort(logits_f)[-top_k:][::-1]
                        
                        global_hit = attr_tok_id in top_k_ids_g
                        family_hit = attr_tok_id in top_k_ids_f
                        
                        if global_hit:
                            fam_global_success += 1
                        if family_hit:
                            fam_family_success += 1
                        fam_total += 1
                        
                    except Exception as e:
                        L.log(f"    Error: {fam}/{test_noun}/{test_attr}: {e}")
                        continue
            
            if fam_total > 0:
                layer_result["global"][fam] = {
                    "success": fam_global_success,
                    "total": fam_total,
                    "rate": fam_global_success / fam_total
                }
                layer_result["family"][fam] = {
                    "success": fam_family_success,
                    "total": fam_total,
                    "rate": fam_family_success / fam_total
                }
                L.log(f"  L{layer} {fam}: global={fam_global_success}/{fam_total}({fam_global_success/fam_total:.1%}), "
                      f"family={fam_family_success}/{fam_total}({fam_family_success/fam_total:.1%})")
        
        results[layer] = layer_result
    
    return results

# ===================== P312: 族间编码差异分析 =====================
def run_p312_family_difference(G_dict, labels_dict, attr_type, attr_list, families, layers, L):
    """分析不同族的CCA方向的余弦相似度"""
    L.log(f"\n=== P312: 族间编码差异 ({attr_type}) ===")
    
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
        
        pca_dim = min(30, N - 1, D)
        pca = PCA(n_components=pca_dim)
        G_pca = pca.fit_transform(G_centered)
        if np.any(np.isnan(G_pca)):
            continue
        
        # 收集每个族的f_base方向和CCA方向
        family_fbase = {}
        family_cca_dirs = {}
        
        for fam in families:
            fam_mask = [i for i, l in enumerate(labels) if l[3] == fam]
            if len(fam_mask) < 12:
                continue
            
            G_fam = G_centered[fam_mask]
            labels_fam = [labels[i] for i in fam_mask]
            
            # f_base: 族的平均偏移
            family_fbase[fam] = np.mean(G_fam, axis=0)
            
            # 族内CCA方向
            pca_dim_f = min(30, len(fam_mask) - 1, D)
            pca_f = PCA(n_components=pca_dim_f)
            G_pca_f = pca_f.fit_transform(G_fam)
            
            if np.any(np.isnan(G_pca_f)):
                continue
            
            Y_fam = build_indicator(labels_fam, attr_list, attr_type)
            n_cca_f = min(5, len(attr_list) - 1, G_pca_f.shape[1] - 1)
            
            try:
                cca_f = CCA(n_components=n_cca_f, max_iter=500)
                cca_f.fit(G_pca_f, Y_fam)
                family_cca_dirs[fam] = cca_f.x_weights_  # (pca_dim_f, n_cca_f)
            except:
                pass
        
        # 计算族间f_base的余弦相似度
        fbase_sims = {}
        fam_list = sorted(family_fbase.keys())
        for i, f1 in enumerate(fam_list):
            for j, f2 in enumerate(fam_list):
                if i >= j:
                    continue
                v1, v2 = family_fbase[f1], family_fbase[f2]
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 > 1e-10 and n2 > 1e-10:
                    cos_sim = np.dot(v1, v2) / (n1 * n2)
                else:
                    cos_sim = 0.0
                fbase_sims[f"{f1}_vs_{f2}"] = cos_sim
        
        # 计算族间CCA方向的子空间重叠
        cca_overlaps = {}
        for i, f1 in enumerate(fam_list):
            for j, f2 in enumerate(fam_list):
                if i >= j:
                    continue
                if f1 not in family_cca_dirs or f2 not in family_cca_dirs:
                    continue
                
                U1, U2 = family_cca_dirs[f1], family_cca_dirs[f2]
                # 子空间重叠 = ||U1^T U2||_F / sqrt(||U1||_F * ||U2||_F)
                # 但U1和U2维度可能不同, 取前min(n1,n2)列
                k = min(U1.shape[1], U2.shape[1])
                U1_k, U2_k = U1[:, :k], U2[:, :k]
                
                # 需要对齐维度: 用PCA投影到公共空间
                if U1_k.shape[0] != U2_k.shape[0]:
                    min_dim = min(U1_k.shape[0], U2_k.shape[0])
                    U1_k = U1_k[:min_dim, :]
                    U2_k = U2_k[:min_dim, :]
                
                try:
                    overlap = np.linalg.norm(U1_k.T @ U2_k, 'fro') / np.sqrt(
                        np.linalg.norm(U1_k, 'fro') * np.linalg.norm(U2_k, 'fro'))
                except:
                    overlap = 0.0
                
                cca_overlaps[f"{f1}_vs_{f2}"] = overlap
        
        # CCA方向的平均余弦相似度(前3个方向)
        cca_dir_sims = {}
        for i, f1 in enumerate(fam_list):
            for j, f2 in enumerate(fam_list):
                if i >= j:
                    continue
                if f1 not in family_cca_dirs or f2 not in family_cca_dirs:
                    continue
                
                U1, U2 = family_cca_dirs[f1], family_cca_dirs[f2]
                k = min(U1.shape[1], U2.shape[1], 3)
                
                if U1.shape[0] != U2.shape[0]:
                    min_dim = min(U1.shape[0], U2.shape[0])
                    U1 = U1[:min_dim, :]
                    U2 = U2[:min_dim, :]
                
                dir_sims = []
                for d in range(k):
                    n1 = np.linalg.norm(U1[:, d])
                    n2 = np.linalg.norm(U2[:, d])
                    if n1 > 1e-10 and n2 > 1e-10:
                        dir_sims.append(abs(np.dot(U1[:, d], U2[:, d]) / (n1 * n2)))
                
                cca_dir_sims[f"{f1}_vs_{f2}"] = np.mean(dir_sims) if dir_sims else 0.0
        
        results[layer] = {
            "fbase_sims": fbase_sims,
            "cca_overlaps": cca_overlaps,
            "cca_dir_sims": cca_dir_sims,
            "n_families": len(fam_list)
        }
        
        L.log(f"  L{layer} fbase_sims: {', '.join(f'{k}={v:.3f}' for k, v in fbase_sims.items())}")
        L.log(f"  L{layer} cca_overlaps: {', '.join(f'{k}={v:.3f}' for k, v in cca_overlaps.items())}")
        L.log(f"  L{layer} cca_dir_sims: {', '.join(f'{k}={v:.3f}' for k, v in cca_dir_sims.items())}")
    
    return results

# ===================== P313: 通用基底+族特异偏移 =====================
def run_p313_universal_plus_specific(G_dict, labels_dict, attr_type, attr_list, families, layers, L):
    """混合模型: φ_i^(k) = φ_i^(global) + Δφ_i^(k)"""
    L.log(f"\n=== P313: 通用基底+族特异偏移 ({attr_type}) ===")
    
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
        
        pca_dim = min(30, N - 1, D)
        pca = PCA(n_components=pca_dim)
        G_pca = pca.fit_transform(G_centered)
        if np.any(np.isnan(G_pca)):
            continue
        
        # 全局CCA方向(通用基底)
        Y_all = build_indicator(labels, attr_list, attr_type)
        n_cca = min(10, len(attr_list) - 1, G_pca.shape[1] - 1)
        
        try:
            cca_global = CCA(n_components=n_cca, max_iter=500)
            cca_global.fit(G_pca, Y_all)
            U_global = cca_global.x_weights_  # (pca_dim, n_cca) — 通用基底
        except:
            continue
        
        # 每个族的CCA方向
        family_contributions = {}
        
        for fam in families:
            fam_mask = [i for i, l in enumerate(labels) if l[3] == fam]
            if len(fam_mask) < 12:
                continue
            
            G_fam = G_centered[fam_mask]
            labels_fam = [labels[i] for i in fam_mask]
            
            # 族的属性均值(在G_centered空间)
            fam_attr_means = {}
            for attr in attr_list:
                attr_mask_f = [i for i, l in enumerate(labels_fam) if l[2] == attr]
                if len(attr_mask_f) < 2:
                    continue
                fam_attr_means[attr] = np.mean(G_fam[attr_mask_f], axis=0)
            
            if len(fam_attr_means) < 3:
                continue
            
            # 构造族特异的"算子矩阵"
            attr_names_f = sorted(fam_attr_means.keys())
            fam_ops = np.array([fam_attr_means[a] for a in attr_names_f])  # (K_f, D)
            
            # 同样构造全局算子矩阵(用相同属性)
            global_attr_means = {}
            for attr in attr_names_f:
                attr_mask_g = [i for i, l in enumerate(labels) if l[2] == attr]
                if len(attr_mask_g) < 2:
                    continue
                global_attr_means[attr] = np.mean(G_centered[attr_mask_g], axis=0)
            
            if len(global_attr_means) < 3:
                continue
            
            global_ops = np.array([global_attr_means[a] for a in attr_names_f])  # (K_f, D)
            
            # 分解: fam_ops = global_ops + delta_ops
            delta_ops = fam_ops - global_ops
            
            # 计算方差贡献比
            var_global = np.sum(global_ops ** 2)
            var_delta = np.sum(delta_ops ** 2)
            var_total = var_global + var_delta
            
            if var_total < 1e-10:
                continue
            
            global_ratio = var_global / var_total
            delta_ratio = var_delta / var_total
            
            # PCA分析delta_ops的结构
            if delta_ops.shape[0] >= 2:
                pca_delta = PCA()
                pca_delta.fit(delta_ops)
                delta_eigenvalues = pca_delta.explained_variance_ratio_[:5]
            else:
                delta_eigenvalues = []
            
            # 全局算子的CCA投影
            try:
                # 将全局算子和族算子投影到全局CCA空间
                global_ops_pca = pca.transform(global_ops)  # (K_f, pca_dim)
                fam_ops_pca = pca.transform(fam_ops)  # (K_f, pca_dim)
                delta_ops_pca = fam_ops_pca - global_ops_pca
                
                # 在CCA空间中的方差比
                var_g_cca = np.sum(global_ops_pca ** 2)
                var_d_cca = np.sum(delta_ops_pca ** 2)
                total_cca = var_g_cca + var_d_cca
                if total_cca > 1e-10:
                    global_ratio_cca = var_g_cca / total_cca
                    delta_ratio_cca = var_d_cca / total_cca
                else:
                    global_ratio_cca = 1.0
                    delta_ratio_cca = 0.0
            except:
                global_ratio_cca = global_ratio
                delta_ratio_cca = delta_ratio
            
            family_contributions[fam] = {
                "global_ratio": global_ratio,
                "delta_ratio": delta_ratio,
                "global_ratio_cca": global_ratio_cca,
                "delta_ratio_cca": delta_ratio_cca,
                "delta_eigenvalues": delta_eigenvalues.tolist() if len(delta_eigenvalues) > 0 else [],
                "n_attrs": len(attr_names_f)
            }
            
            L.log(f"  L{layer} {fam}: global_ratio={global_ratio:.3f}, delta_ratio={delta_ratio:.3f}, "
                  f"cca: global={global_ratio_cca:.3f}, delta={delta_ratio_cca:.3f}")
        
        # 汇总
        if family_contributions:
            avg_global = np.mean([v["global_ratio"] for v in family_contributions.values()])
            avg_delta = np.mean([v["delta_ratio"] for v in family_contributions.values()])
            avg_global_cca = np.mean([v["global_ratio_cca"] for v in family_contributions.values()])
            avg_delta_cca = np.mean([v["delta_ratio_cca"] for v in family_contributions.values()])
            
            results[layer] = {
                "families": family_contributions,
                "avg_global_ratio": avg_global,
                "avg_delta_ratio": avg_delta,
                "avg_global_ratio_cca": avg_global_cca,
                "avg_delta_ratio_cca": avg_delta_cca
            }
            
            L.log(f"  L{layer} AVG: global={avg_global:.3f}, delta={avg_delta:.3f}, "
                  f"cca: global={avg_global_cca:.3f}, delta={avg_delta_cca:.3f}")
    
    return results

# ===================== P314: 大规模五族泛化测试 =====================
def run_p314_cross_family_intervention(model, tokenizer, device, G_dict, labels_dict,
                                        attr_type, attr_list, families, test_layers, L):
    """5×5族间干预转移矩阵"""
    L.log(f"\n=== P314: 大规模五族泛化测试 ({attr_type}) ===")
    
    results = {}
    
    for layer in test_layers:
        G = G_dict[layer]
        labels = labels_dict[layer] if isinstance(labels_dict, dict) else labels_dict
        N, D = G.shape
        
        if N < 30:
            continue
        
        G_centered, noun_means = noun_centered_G(G, labels)
        if np.any(np.isnan(G_centered)) or np.all(G_centered == 0):
            continue
        
        # 为每个族训练算子
        family_operators = {}
        for fam in families:
            fam_mask = [i for i, l in enumerate(labels) if l[3] == fam]
            if len(fam_mask) < 12:
                continue
            
            G_fam = G_centered[fam_mask]
            labels_fam = [labels[i] for i in fam_mask]
            
            ops = {}
            for attr in attr_list[:4]:  # 前4个属性
                attr_mask_f = [i for i, l in enumerate(labels_fam) if l[2] == attr]
                if len(attr_mask_f) < 2:
                    continue
                ops[attr] = np.mean(G_fam[attr_mask_f], axis=0)
            
            family_operators[fam] = ops
        
        # 构建转移矩阵: source_family × target_family
        transfer_matrix = {}
        
        for source_fam in family_operators:
            transfer_matrix[source_fam] = {}
            
            for target_fam in families:
                target_nouns = STIMULI[target_fam]
                # 选3个测试名词
                test_nouns = target_nouns[::4][:3]  # 均匀采样
                test_attrs = list(family_operators[source_fam].keys())[:3]
                
                success = 0
                total = 0
                
                for test_noun in test_nouns:
                    for test_attr in test_attrs:
                        try:
                            # 获取noun基线
                            noun_phrase = f"a {test_noun}"
                            toks = tokenizer(noun_phrase, return_tensors="pt").to(device)
                            with torch.no_grad():
                                out = model(toks.input_ids, output_hidden_states=True)
                                h_noun = out.hidden_states[layer][0, -1].float().cpu().numpy()
                            
                            # 用source族的算子干预target族的名词
                            if test_attr in family_operators[source_fam]:
                                h_new = h_noun + family_operators[source_fam][test_attr]
                            else:
                                continue
                            
                            with torch.no_grad():
                                h_tensor = torch.tensor(h_new, dtype=torch.bfloat16).unsqueeze(0).unsqueeze(0).to(device)
                                lm_head = model.lm_head if hasattr(model, 'lm_head') else model.get_output_embeddings()
                                logits = lm_head(h_tensor)[0, 0].float().cpu().numpy()
                            
                            # 检查目标属性词是否在top-20
                            attr_tok_id = tokenizer.encode(f" {test_attr}", add_special_tokens=False)
                            if len(attr_tok_id) == 0:
                                continue
                            attr_tok_id = attr_tok_id[0]
                            
                            top_k = 20
                            top_k_ids = np.argsort(logits)[-top_k:][::-1]
                            
                            if attr_tok_id in top_k_ids:
                                success += 1
                            total += 1
                            
                        except Exception as e:
                            continue
                
                if total > 0:
                    transfer_matrix[source_fam][target_fam] = {
                        "success": success,
                        "total": total,
                        "rate": success / total
                    }
        
        results[layer] = transfer_matrix
        
        # 打印转移矩阵
        L.log(f"  L{layer} Transfer Matrix:")
        source_fams = sorted(family_operators.keys())
        target_fams = sorted(families)
        header = "Source\\Target\t" + "\t".join([f.replace("_family","")[:4] for f in target_fams])
        L.log(f"  {header}")
        for sf in source_fams:
            row = f"  {sf.replace('_family','')[:4]}\t\t"
            for tf in target_fams:
                if sf in transfer_matrix and tf in transfer_matrix[sf]:
                    rate = transfer_matrix[sf][tf]["rate"]
                    row += f"{rate:.1%}\t\t"
                else:
                    row += "N/A\t\t"
            L.log(row)
    
    return results

# ===================== 主函数 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3", choices=["qwen3","glm4","deepseek7b"])
    args = parser.parse_args()
    
    model_name = args.model
    L.log(f"=== Phase LIV: 族条件化建模 ({model_name}) ===")
    L.log(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 加载模型
    L.log("Loading model...")
    model, tokenizer, device = load_model(model_name)
    n_layers = len(model.model.layers) if hasattr(model, 'model') and hasattr(model.model, 'layers') else 0
    L.log(f"Model loaded: {n_layers} layers, device={device}")
    
    # 选择测试层
    if n_layers > 0:
        last_layer = n_layers - 1
        test_layers = [last_layer // 3, last_layer // 2, 2 * last_layer // 3, last_layer]
    else:
        test_layers = [16, 24, 32, 40]
    
    L.log(f"Test layers: {test_layers}")
    
    # 构建三元组
    all_triples = []
    all_labels = []
    for n in ALL_NOUNS:
        for c in STIMULI["color_attrs"]:
            all_triples.append((n, c, f"{c} {n}"))
            all_labels.append((n, "color", c, NOUN_TO_FAMILY.get(n, "unknown")))
        for t in STIMULI["taste_attrs"]:
            all_triples.append((n, t, f"{t} {n}"))
            all_labels.append((n, "taste", t, NOUN_TO_FAMILY.get(n, "unknown")))
        for s in STIMULI["size_attrs"]:
            all_triples.append((n, s, f"{s} {n}"))
            all_labels.append((n, "size", s, NOUN_TO_FAMILY.get(n, "unknown")))
    
    L.log(f"Total triples: {len(all_triples)}")
    
    # 收集所有层的残差
    L.log("Collecting residuals...")
    G_all_layers = collect_residuals_all_layers(model, tokenizer, device, 
                                                 PROMPT_TEMPLATES_30, all_triples, test_layers)
    
    # 为每个属性类型分别运行实验
    all_results = {}
    
    for attr_type, attr_list in [("color", STIMULI["color_attrs"]), 
                                  ("taste", STIMULI["taste_attrs"]),
                                  ("size", STIMULI["size_attrs"])]:
        L.log(f"\n{'='*60}")
        L.log(f"Processing {attr_type}...")
        
        # 筛选当前属性类型的标签
        attr_labels = [l for l in all_labels if l[1] == attr_type]
        attr_triples = [(l[0], l[2], f"{l[2]} {l[0]}") for l in attr_labels]
        
        # 重新收集当前属性类型的G项
        L.log(f"Collecting {attr_type} G terms...")
        G_attr_layers = collect_residuals_all_layers(model, tokenizer, device,
                                                      PROMPT_TEMPLATES_30, attr_triples, test_layers)
        
        # P310: 族条件化CCA
        p310 = run_p310_family_cca(G_attr_layers, attr_labels, attr_type, attr_list, 
                                    FAMILY_NAMES, test_layers, L)
        all_results[f"p310_{attr_type}"] = p310
        
        # P312: 族间编码差异
        p312 = run_p312_family_difference(G_attr_layers, attr_labels, attr_type, attr_list,
                                           FAMILY_NAMES, test_layers, L)
        all_results[f"p312_{attr_type}"] = p312
        
        # P313: 通用基底+族特异偏移
        p313 = run_p313_universal_plus_specific(G_attr_layers, attr_labels, attr_type, attr_list,
                                                 FAMILY_NAMES, test_layers, L)
        all_results[f"p313_{attr_type}"] = p313
    
    # P311和P314需要干预实验, 只在末层运行
    last_l = test_layers[-1]
    L.log(f"\n{'='*60}")
    L.log("Running intervention experiments (last layer only)...")
    
    for attr_type, attr_list in [("color", STIMULI["color_attrs"]),
                                  ("taste", STIMULI["taste_attrs"]),
                                  ("size", STIMULI["size_attrs"])]:
        attr_labels = [l for l in all_labels if l[1] == attr_type]
        attr_triples = [(l[0], l[2], f"{l[2]} {l[0]}") for l in attr_labels]
        
        G_attr = {last_l: G_all_layers[last_l][[i for i, l in enumerate(all_labels) if l[1] == attr_type]]}
        labels_attr = {last_l: attr_labels}
        
        # P311: 族内干预
        p311 = run_p311_family_intervention(model, tokenizer, device, G_attr, labels_attr,
                                             attr_type, attr_list, FAMILY_NAMES, [last_l], L)
        all_results[f"p311_{attr_type}"] = p311
        
        # P314: 跨族泛化
        p314 = run_p314_cross_family_intervention(model, tokenizer, device, G_attr, labels_attr,
                                                    attr_type, attr_list, FAMILY_NAMES, [last_l], L)
        all_results[f"p314_{attr_type}"] = p314
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    out_file = OUT_DIR / f"phase_liv_p310_314_{model_name}_{timestamp}.json"
    
    # 将numpy数组转为列表
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
        return obj
    
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(convert(all_results), f, indent=2, ensure_ascii=False)
    
    L.log(f"\nResults saved to {out_file}")
    L.log(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.close()
    
    # 清理GPU内存
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"\n{'='*60}")
    print(f"Phase LIV ({model_name}) completed!")
    print(f"Results: {out_file}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
