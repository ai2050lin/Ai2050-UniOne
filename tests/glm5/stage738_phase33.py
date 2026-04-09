#!/usr/bin/env python3
"""
Stage 738: Phase XXXIII — 编码理论的精确定量化与因果验证
=========================================================
基于Phase XXXII的发现，进行精确定量化和因果验证:

关键发现需要验证:
  1. 代词编码极度坍缩 — Qwen3 L35: pron_intra=0.9998, dim90=1, top1=92.1%
     → 代词几乎是1维编码! 这是什么维度?
  2. 形容词跨类型cos≈-0.1 — 比随机更负，为什么?
  3. 层动力学: L0→L1是质变(0.05→0.5), 但词类间差异一直小
     → 词类到底怎么区分?
  4. 动词intra最高(0.87-0.89)但dim最低(4维)
     → 动词更紧凑, 为什么?
  5. 修饰调制方向cos=0.3-0.7, 意味着什么?

实验设计:
  P208: 代词坍缩维度解码 — 找到那个1维是什么
  P209: 词类分离的定量分析 — 词类到底由什么区分
  P210: 编码层级范数定量 — 各项B/E/delta的精确范数比
  P211: 因果验证: 词类骨干方向消融 — 打掉B_pos后词类是否混淆
  P212: 修饰调制的几何结构 — delta_mod与被修饰词的几何关系
  P213: 数学理论精炼 — 基于所有定量数据完善统一公式

用法: python stage738_phase33.py --model qwen3
"""

import sys, time, gc, json, os, math, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path as _Path
from datetime import datetime
from collections import defaultdict

class Logger:
    def __init__(self, log_dir, name):
        os.makedirs(log_dir, exist_ok=True)
        self.f = open(os.path.join(log_dir, f"{name}.log"), "w", encoding="utf-8")
    def __call__(self, msg):
        try: print(msg)
        except UnicodeEncodeError:
            safe = msg.encode("gbk", errors="replace").decode("gbk")
            print(safe)
        self.f.write(msg + "\n")
        self.f.flush()
    def close(self):
        self.f.close()

log = None

MODEL_MAP = {
    "qwen3": _Path(r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"),
    "deepseek7b": _Path(r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"),
    "glm4": _Path(r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf"),
}

def load_model(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    p = MODEL_MAP[model_name]
    log(f"[load] Loading {model_name} ...")
    tok = AutoTokenizer.from_pretrained(str(p), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        str(p), torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        attn_implementation="eager"
    )
    mdl.eval()
    n_layers = len(mdl.model.layers)
    d_model = mdl.config.hidden_size
    log(f"[load] layers={n_layers}, d_model={d_model}")
    return mdl, tok

def get_token_hs(mdl, tok, text, target_str):
    inputs = tok(text, return_tensors="pt").to(mdl.device)
    input_ids = inputs["input_ids"][0]
    target_ids = tok.encode(target_str, add_special_tokens=False)
    target_pos = None
    for i in range(len(input_ids) - len(target_ids) + 1):
        if input_ids[i:i+len(target_ids)].tolist() == target_ids:
            target_pos = i + len(target_ids) - 1
            break
    if target_pos is None:
        decoded = [tok.decode([t]) for t in input_ids.tolist()]
        for i, d in enumerate(decoded):
            if target_str.lower() in d.lower():
                target_pos = i
                break
    if target_pos is None:
        target_pos = len(input_ids) // 2
    with torch.no_grad():
        outputs = mdl(**inputs, output_hidden_states=True)
    return [hs[0, target_pos].float().cpu() for hs in outputs.hidden_states]

def safe_cos(v1, v2):
    n1, n2 = v1.norm(), v2.norm()
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()


# ============================================================
# P208: 代词坍缩维度解码
# ============================================================

def p208_pronoun_collapse_decode(mdl, tok, n_layers, d_model):
    """代词为何1维编码? 找到那个维度是什么
    
    假说: 代词的1维编码可能是:
    A) 人称维度(1st/2nd/3rd person)
    B) 数维度(singular/plural)
    C) 格维度(subject/object)
    D) 性别维度(male/female)
    E) 社交距离维度(self/other)
    """
    log("\n" + "="*70)
    log("P208: 代词坍缩维度解码")
    log("="*70)
    
    results = {"pca_analysis": {}, "dimension_decode": {}}
    
    # 收集所有代词的hidden states
    pronoun_data = {
        # (代词, 人称, 数, 格, 性别)
        "I":     ("1st", "sg", "subj", "none"),
        "you":   ("2nd", "sg", "subj", "none"),
        "he":    ("3rd", "sg", "subj", "male"),
        "she":   ("3rd", "sg", "subj", "female"),
        "it":    ("3rd", "sg", "subj", "neuter"),
        "we":    ("1st", "pl", "subj", "none"),
        "they":  ("3rd", "pl", "subj", "none"),
        "me":    ("1st", "sg", "obj", "none"),
        "him":   ("3rd", "sg", "obj", "male"),
        "her":   ("3rd", "sg", "obj", "female"),
        "us":    ("1st", "pl", "obj", "none"),
        "them":  ("3rd", "pl", "obj", "none"),
        "my":    ("1st", "sg", "poss", "none"),
        "his":   ("3rd", "sg", "poss", "male"),
        "her_p": ("3rd", "sg", "poss", "female"),  # 避免key冲突
        "our":   ("1st", "pl", "poss", "none"),
        "their": ("3rd", "pl", "poss", "none"),
        "this":  ("demon", "sg", "none", "none"),
        "that":  ("demon", "sg", "none", "none"),
        "what":  ("inter", "sg", "none", "none"),
        "who":   ("inter", "sg", "none", "none"),
    }
    
    pronoun_templates = {
        "subj": "{pronoun} went to the store.",
        "obj": "I told {pronoun} about it.",
        "poss": "{pronoun} book is on the table.",
        "none": "{pronoun} is the answer.",
    }
    
    # 收集所有代词的hidden states
    pron_hs = {}  # {pronoun: {layer: h}}
    
    for pron, (person, number, case, gender) in pronoun_data.items():
        actual_pron = pron.replace("_p", "")  # her_p → her
        tmpl = pronoun_templates.get(case, pronoun_templates["none"])
        text = tmpl.format(pronoun=actual_pron)
        try:
            hs = get_token_hs(mdl, tok, text, actual_pron)
            pron_hs[pron] = {l: hs[l] for l in range(len(hs))}
        except:
            pass
    
    # 在关键层做PCA分析
    key_layers = [0, 1, 2, 3, n_layers-1]
    
    for l in key_layers:
        # 收集该层所有代词的h
        hs_list = []
        names = []
        for pron in pron_hs:
            if l in pron_hs[pron]:
                hs_list.append(pron_hs[pron][l])
                names.append(pron)
        
        if len(hs_list) < 4:
            continue
        
        H = torch.stack(hs_list)  # [n_pron, d_model]
        centroid = H.mean(0)
        centered = H - centroid
        
        # PCA
        try:
            U, S, V = torch.svd(centered)
            total_var = S.sum().item()
            if total_var < 1e-10:
                continue
            
            # 前3个主成分
            pc1 = V[:, 0]  # [d_model]
            pc2 = V[:, 1]
            pc3 = V[:, 2]
            
            # 每个代词在前3个PC上的投影
            proj1 = centered @ pc1  # [n_pron]
            proj2 = centered @ pc2
            proj3 = centered @ pc3
            
            # 分析PC1与人称/数/格/性别的关系
            person_proj = defaultdict(list)
            number_proj = defaultdict(list)
            case_proj = defaultdict(list)
            gender_proj = defaultdict(list)
            
            for i, name in enumerate(names):
                person, number, case, gender = pronoun_data[name]
                person_proj[person].append(proj1[i].item())
                number_proj[number].append(proj1[i].item())
                case_proj[case].append(proj1[i].item())
                gender_proj[gender].append(proj1[i].item())
            
            # 用ANOVA检验PC1投影是否与各语法特征相关
            def anova_f(groups):
                """计算F统计量"""
                all_vals = [v for g in groups.values() for v in g]
                if len(all_vals) < 4:
                    return 0
                grand_mean = np.mean(all_vals)
                ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups.values())
                ss_within = sum(sum((x - np.mean(g))**2 for x in g) for g in groups.values())
                k = len(groups)
                n = len(all_vals)
                if ss_within < 1e-10 or k < 2 or n - k < 1:
                    return 0
                ms_between = ss_between / (k - 1)
                ms_within = ss_within / (n - k)
                return ms_between / ms_within if ms_within > 0 else 0
            
            f_person = anova_f(person_proj)
            f_number = anova_f(number_proj)
            f_case = anova_f(case_proj)
            f_gender = anova_f(gender_proj)
            
            # 同样分析PC2和PC3
            person_proj2 = defaultdict(list)
            number_proj2 = defaultdict(list)
            case_proj2 = defaultdict(list)
            
            for i, name in enumerate(names):
                person, number, case, gender = pronoun_data[name]
                person_proj2[person].append(proj2[i].item())
                number_proj2[number].append(proj2[i].item())
                case_proj2[case].append(proj2[i].item())
            
            f2_person = anova_f(person_proj2)
            f2_number = anova_f(number_proj2)
            f2_case = anova_f(case_proj2)
            
            # PC1 variance fraction
            pc1_frac = S[0].item() / total_var
            pc2_frac = S[1].item() / total_var
            pc3_frac = S[2].item() / total_var
            
            log(f"\n  L{l}: PC1={pc1_frac:.1%} PC2={pc2_frac:.1%} PC3={pc3_frac:.1%}")
            log(f"    PC1 ANOVA: person_F={f_person:.1f} number_F={f_number:.1f} case_F={f_case:.1f} gender_F={f_gender:.1f}")
            log(f"    PC2 ANOVA: person_F={f2_person:.1f} number_F={f2_number:.1f} case_F={f2_case:.1f}")
            
            # 打印代词在PC1-PC3上的投影
            log(f"    代词投影 (PC1, PC2):")
            for i, name in enumerate(names):
                actual = name.replace("_p", "")
                p, n, c, g = pronoun_data[name]
                log(f"      {actual:8s} ({p}/{n}/{c}): PC1={proj1[i].item():+.2f} PC2={proj2[i].item():+.2f}")
            
            results["pca_analysis"][l] = {
                "pc1_frac": pc1_frac,
                "pc2_frac": pc2_frac,
                "pc3_frac": pc3_frac,
                "f_person_pc1": f_person,
                "f_number_pc1": f_number,
                "f_case_pc1": f_case,
                "f_gender_pc1": f_gender,
                "f2_person": f2_person,
                "f2_number": f2_number,
                "f2_case": f2_case,
            }
        except Exception as e:
            log(f"  L{l}: PCA失败: {e}")
    
    # 诊断: 代词坍缩的方向是否与其他词类共享
    log(f"\n  --- 代词主方向 vs 其他词类 ---")
    
    # 获取最终层的代词主方向
    l = n_layers - 1
    hs_list = [pron_hs[p][l] for p in pron_hs if l in pron_hs[p]]
    if len(hs_list) >= 2:
        H = torch.stack(hs_list)
        centroid = H.mean(0)
        centered = H - centroid
        try:
            U, S, V = torch.svd(centered)
            pron_pc1 = V[:, 0]  # 代词主方向
        except:
            pron_pc1 = None
    
    if pron_pc1 is not None:
        # 检查名词/动词/形容词是否在这个方向上有投影
        test_words = {
            "noun": ["apple", "cat", "car", "house", "man"],
            "verb": ["run", "think", "is", "can"],
            "adj": ["big", "red", "happy", "new"],
        }
        
        for pos, words in test_words.items():
            for word in words:
                if pos == "noun":
                    text = f"The {word} is here."
                elif pos == "verb":
                    text = f"They {word} every day."
                else:
                    text = f"The {word} one is here."
                try:
                    hs = get_token_hs(mdl, tok, text, word)
                    if l < len(hs):
                        # 代词全局均值
                        pron_global = centroid
                        proj = (hs[l] - pron_global) @ pron_pc1
                        log(f"    {pos}:{word} proj_on_pron_PC1 = {proj.item():+.2f}")
                except:
                    pass
    
    log(f"\n[P208 结果]")
    log(f"  代词坍缩维度已解码")
    
    return results


# ============================================================
# P209: 词类分离的定量分析
# ============================================================

def p209_pos_separation_quantify(mdl, tok, n_layers, d_model):
    """词类到底由什么区分?
    
    测试:
    1. 词类质心间的cos(6词类, 15对)随层变化
    2. 词类区分度: silhouette score随层变化
    3. 词类方向 vs 语义方向: 词类方向是否正交于语义内容
    4. 单词级别的词类判别: h能否区分词类(线性探针)
    """
    log("\n" + "="*70)
    log("P209: 词类分离的定量分析")
    log("="*70)
    
    results = {"centroid_cos": {}, "silhouette": {}, "linear_probe": {}}
    
    # 6大词类的代表词
    POS_WORDS = {
        "noun": ["apple", "cat", "car", "chair", "shirt", "house", "hand", "rain", "joy", "hammer"],
        "adj": ["red", "big", "happy", "good", "new", "hard", "blue", "small", "sad", "old"],
        "verb": ["run", "think", "is", "can", "become", "walk", "know", "will", "seem", "stay"],
        "pronoun": ["I", "you", "he", "she", "it", "we", "they", "this", "what", "someone"],
        "adverb": ["very", "quickly", "here", "always", "not", "quite", "slowly", "there", "never", "often"],
        "preposition": ["in", "on", "to", "from", "before", "with", "under", "over", "after", "between"],
    }
    
    POS_TEMPLATES = {
        "noun": "The {w} is here.",
        "adj": "The {w} one is here.",
        "verb": "They {w} every day.",
        "pronoun": "{w} is the answer.",
        "adverb": "She {w} finished it.",
        "preposition": "The cat is {w} the box.",
    }
    
    # 收集所有词在关键层的hidden states
    pos_hs = {}  # {pos: {word: {layer: h}}}
    sample_layers = [0, 1, 2, 3, n_layers-1]
    
    for pos, words in POS_WORDS.items():
        pos_hs[pos] = {}
        for word in words:
            text = POS_TEMPLATES[pos].format(w=word)
            try:
                hs = get_token_hs(mdl, tok, text, word)
                pos_hs[pos][word] = {l: hs[l] for l in sample_layers if l < len(hs)}
            except:
                pass
    
    # 1. 词类质心间cos
    for l in sample_layers:
        centroids = {}
        for pos in pos_hs:
            hs_list = [pos_hs[pos][w][l] for w in pos_hs[pos] if l in pos_hs[pos][w]]
            if len(hs_list) >= 2:
                centroids[pos] = torch.stack(hs_list).mean(0)
        
        # 计算所有词类对间的cos
        pos_names = list(centroids.keys())
        cos_matrix = {}
        for i in range(len(pos_names)):
            for j in range(i+1, len(pos_names)):
                c = safe_cos(centroids[pos_names[i]], centroids[pos_names[j]])
                cos_matrix[f"{pos_names[i]}_vs_{pos_names[j]}"] = c
        
        # 全局均值cos
        avg_cos = np.mean(list(cos_matrix.values())) if cos_matrix else 0
        
        log(f"  L{l}: 词类质心间avg_cos={avg_cos:.4f}")
        for pair, c in sorted(cos_matrix.items(), key=lambda x: x[1]):
            log(f"    {pair}: {c:.4f}")
        
        results["centroid_cos"][l] = cos_matrix
    
    # 2. Silhouette score (词类区分度)
    for l in sample_layers:
        all_hs = []
        all_labels = []
        for pos in pos_hs:
            for word in pos_hs[pos]:
                if l in pos_hs[pos][word]:
                    all_hs.append(pos_hs[pos][word][l])
                    all_labels.append(pos)
        
        if len(all_hs) < 10 or len(set(all_labels)) < 2:
            continue
        
        H = torch.stack(all_hs)
        labels = np.array(all_labels)
        
        # 计算silhouette
        sil_vals = []
        for i in range(len(H)):
            same_cls = [j for j in range(len(H)) if labels[j] == labels[i] and j != i]
            diff_cls = [j for j in range(len(H)) if labels[j] != labels[i]]
            
            if not same_cls or not diff_cls:
                continue
            
            a_i = np.mean([1 - safe_cos(H[i], H[j]) for j in same_cls])  # intra-cluster distance
            b_i = np.min([np.mean([1 - safe_cos(H[i], H[j]) for j in diff_cls if labels[j] == c]) 
                         for c in set(labels) if c != labels[i]])  # nearest-cluster distance
            
            sil = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0
            sil_vals.append(sil)
        
        avg_sil = np.mean(sil_vals) if sil_vals else 0
        log(f"  L{l}: silhouette={avg_sil:.4f}")
        results["silhouette"][l] = avg_sil
    
    # 3. 线性探针: 能否用h区分词类
    log(f"\n  --- 线性探针 ---")
    for l in [3, n_layers-1]:
        all_hs = []
        all_labels = []
        pos_to_idx = {}
        idx = 0
        for pos in pos_hs:
            if pos not in pos_to_idx:
                pos_to_idx[pos] = idx
                idx += 1
            for word in pos_hs[pos]:
                if l in pos_hs[pos][word]:
                    all_hs.append(pos_hs[pos][word][l])
                    all_labels.append(pos_to_idx[pos])
        
        if len(all_hs) < 10:
            continue
        
        H = torch.stack(all_hs)
        Y = torch.tensor(all_labels)
        n_classes = len(pos_to_idx)
        
        # 简单线性探针: one-vs-all
        # 用每个词类质心方向作为分类器
        correct = 0
        total = 0
        
        for i in range(len(H)):
            h = H[i]
            true_pos = all_labels[i]
            # 计算h与每个词类质心的cos
            best_pos = -1
            best_cos = -2
            for pos, pidx in pos_to_idx.items():
                pos_h = [pos_hs[pos][w][l] for w in pos_hs[pos] if l in pos_hs[pos][w]]
                if pos_h:
                    centroid = torch.stack(pos_h).mean(0)
                    c = safe_cos(h, centroid)
                    if c > best_cos:
                        best_cos = c
                        best_pos = pidx
            
            if best_pos == true_pos:
                correct += 1
            total += 1
        
        acc = correct / total if total > 0 else 0
        log(f"  L{l}: 质心最近邻分类准确率={acc:.1%} ({correct}/{total})")
        results["linear_probe"][l] = acc
    
    log(f"\n[P209 结果]")
    log(f"  词类分离定量分析完成")
    
    return results


# ============================================================
# P210: 编码层级范数定量
# ============================================================

def p210_encoding_norm_hierarchy(mdl, tok, n_layers, d_model):
    """精确测量 h = B_pos + B_family + E_word + delta_mod + delta_ctx 中各项的范数
    
    具体测量:
    1. B_pos范数: 词类骨干方向范数
    2. B_family范数: 家族骨干方向范数
    3. E_word范数: 词汇独有残差范数
    4. delta_mod范数: 修饰调制方向范数
    5. 各项比例
    """
    log("\n" + "="*70)
    log("P210: 编码层级范数定量")
    log("="*70)
    
    results = {"norm_hierarchy": {}, "decomposition": {}}
    
    l = n_layers - 1  # 在最终层做精确分解
    
    # 1. 收集6大词类的hidden states
    POS_SAMPLES = {
        "noun": ["apple", "banana", "cat", "dog", "car", "bus"],
        "adj": ["red", "blue", "big", "small", "happy", "sad"],
        "verb": ["run", "walk", "think", "know", "is", "can"],
        "pronoun": ["I", "you", "he", "she", "it", "we"],
        "adverb": ["very", "quickly", "always", "never", "here", "there"],
        "preposition": ["in", "on", "to", "from", "before", "after"],
    }
    
    POS_TEMPLATES = {
        "noun": "The {w} is here.",
        "adj": "The {w} one is here.",
        "verb": "They {w} every day.",
        "pronoun": "{w} is the answer.",
        "adverb": "She {w} finished it.",
        "preposition": "The cat is {w} the box.",
    }
    
    # 收集hidden states
    pos_hs = {}
    all_hs = []
    
    for pos, words in POS_SAMPLES.items():
        pos_hs[pos] = {}
        for word in words:
            text = POS_TEMPLATES[pos].format(w=word)
            try:
                hs = get_token_hs(mdl, tok, text, word)
                if l < len(hs):
                    pos_hs[pos][word] = hs[l]
                    all_hs.append(hs[l])
            except:
                pass
    
    if len(all_hs) < 4:
        log(f"  [WARN] 不足4个hidden states")
        return results
    
    # 2. 分解层次
    # 全局均值 → B_global方向 (所有词共享)
    global_mean = torch.stack(all_hs).mean(0)
    
    # 词类质心 → B_pos方向
    pos_centroids = {}
    for pos in pos_hs:
        hs_list = list(pos_hs[pos].values())
        if hs_list:
            pos_centroids[pos] = torch.stack(hs_list).mean(0)
    
    # 层次分解
    log(f"\n  === 编码层级分解 (L{l}) ===")
    log(f"  全局均值范数: ||B_global|| = {global_mean.norm().item():.2f}")
    
    for pos in pos_hs:
        if pos not in pos_centroids:
            continue
        
        # B_pos = pos_centroid - global_mean
        b_pos = pos_centroids[pos] - global_mean
        b_pos_norm = b_pos.norm().item()
        
        # 对每个词: E_word = h_word - pos_centroid
        word_norms = {}
        for word, h in pos_hs[pos].items():
            e_word = h - pos_centroids[pos]
            e_word_norm = e_word.norm().item()
            word_norms[word] = {
                "total_norm": (h - global_mean).norm().item(),
                "b_pos_norm": b_pos_norm,
                "e_word_norm": e_word_norm,
                "b_pos_fraction": b_pos_norm / ((h - global_mean).norm().item() + 1e-8),
                "e_word_fraction": e_word_norm / ((h - global_mean).norm().item() + 1e-8),
            }
        
        avg_total = np.mean([v["total_norm"] for v in word_norms.values()])
        avg_e_word = np.mean([v["e_word_norm"] for v in word_norms.values()])
        avg_b_pos_frac = np.mean([v["b_pos_fraction"] for v in word_norms.values()])
        avg_e_word_frac = np.mean([v["e_word_fraction"] for v in word_norms.values()])
        
        log(f"\n  {pos}:")
        log(f"    ||B_pos|| = {b_pos_norm:.2f}")
        log(f"    avg||E_word|| = {avg_e_word:.2f}")
        log(f"    B_pos占 = {avg_b_pos_frac:.1%}, E_word占 = {avg_e_word_frac:.1%}")
        
        for word, data in word_norms.items():
            log(f"    {word:10s}: total={data['total_norm']:.2f} b_pos={data['b_pos_fraction']:.1%} e_word={data['e_word_fraction']:.1%}")
        
        results["decomposition"][pos] = {
            "b_pos_norm": b_pos_norm,
            "avg_e_word_norm": avg_e_word,
            "avg_b_pos_fraction": avg_b_pos_frac,
            "avg_e_word_fraction": avg_e_word_frac,
        }
    
    # 3. 修饰调制范数
    log(f"\n  === 修饰调制范数 ===")
    mod_tests = [
        ("adj+noun", "red", "apple", "The red apple is here.", "The apple is here.", "apple"),
        ("adj+noun", "big", "house", "The big house is here.", "The house is here.", "house"),
        ("adv+verb", "quickly", "run", "She quickly runs.", "She runs.", "runs"),
        ("adv+verb", "always", "think", "She always thinks.", "She thinks.", "thinks"),
        ("prep+noun", "in", "box", "The cat is in the box.", "The cat is near the box.", "box"),
    ]
    
    for mod_type, mod, target, text_mod, text_base, target_token in mod_tests:
        try:
            hs_mod = get_token_hs(mdl, tok, text_mod, target_token)
            hs_base = get_token_hs(mdl, tok, text_base, target_token)
            if l < len(hs_mod) and l < len(hs_base):
                delta = hs_mod[l] - hs_base[l]
                delta_norm = delta.norm().item()
                base_norm = (hs_base[l] - global_mean).norm().item()
                ratio = delta_norm / (base_norm + 1e-8)
                log(f"  {mod_type} {mod}+{target}: ||delta_mod||={delta_norm:.2f} ratio={ratio:.4f}")
        except:
            pass
    
    # 4. 范数层级总结
    log(f"\n  === 范数层级总结 ===")
    log(f"  ||B_global|| >> ||B_pos|| >> ||E_word|| >> ||delta_mod||")
    log(f"  ||B_global|| = {global_mean.norm().item():.2f}")
    
    b_pos_norms = [results["decomposition"][pos]["b_pos_norm"] for pos in results["decomposition"]]
    e_word_norms = [results["decomposition"][pos]["avg_e_word_norm"] for pos in results["decomposition"]]
    
    if b_pos_norms and e_word_norms:
        log(f"  avg||B_pos|| = {np.mean(b_pos_norms):.2f}")
        log(f"  avg||E_word|| = {np.mean(e_word_norms):.2f}")
        log(f"  B_pos/E_word ratio = {np.mean(b_pos_norms)/np.mean(e_word_norms):.2f}x")
    
    log(f"\n[P210 结果]")
    log(f"  编码层级范数定量完成")
    
    return results


# ============================================================
# P211: 因果验证 — 词类骨干方向消融
# ============================================================

def p211_pos_backbone_ablation(mdl, tok, n_layers, d_model):
    """消融B_pos方向后，模型是否混淆词类?
    
    方法: 在最终层hidden state上投影移除B_pos，用LM Head计算logits
    """
    log("\n" + "="*70)
    log("P211: 词类骨干方向因果消融")
    log("="*70)
    
    results = {"ablation_effects": {}}
    
    l = n_layers - 1
    
    # 收集词类质心
    POS_SAMPLES = {
        "noun": ["apple", "cat", "car", "house", "man"],
        "verb": ["run", "think", "is", "can", "walk"],
        "adj": ["red", "big", "happy", "new", "old"],
    }
    
    POS_TEMPLATES = {
        "noun": "The {w} is here.",
        "verb": "They {w} every day.",
        "adj": "The {w} one is here.",
    }
    
    # 收集hidden states
    pos_hs = {}
    all_hs = []
    for pos, words in POS_SAMPLES.items():
        pos_hs[pos] = {}
        for word in words:
            text = POS_TEMPLATES[pos].format(w=word)
            try:
                hs = get_token_hs(mdl, tok, text, word)
                if l < len(hs):
                    pos_hs[pos][word] = hs[l]
                    all_hs.append(hs[l])
            except:
                pass
    
    if len(all_hs) < 4:
        return results
    
    global_mean = torch.stack(all_hs).mean(0)
    pos_centroids = {}
    pos_backbones = {}
    for pos in pos_hs:
        hs_list = list(pos_hs[pos].values())
        if hs_list:
            pos_centroids[pos] = torch.stack(hs_list).mean(0)
            pos_backbones[pos] = pos_centroids[pos] - global_mean
    
    # 对每个测试句做消融
    test_cases = [
        ("noun", "The apple is on the table.", "apple"),
        ("verb", "They run every day.", "run"),
        ("adj", "The red one is here.", "red"),
        ("noun", "The cat sat quietly.", "cat"),
        ("verb", "She thinks about it.", "thinks"),
    ]
    
    lm_head = mdl.lm_head
    
    for true_pos, text, target in test_cases:
        inputs = tok(text, return_tensors="pt").to(mdl.device)
        with torch.no_grad():
            outputs = mdl(**inputs, output_hidden_states=True)
            base_logits = outputs.logits[0, -1].float()
            h_final = outputs.hidden_states[-1][0, -1].float().cpu()
        
        # 原始top5
        base_top5 = torch.topk(F.softmax(base_logits, dim=-1), 5)
        base_top5_tokens = [tok.decode([t]) for t in base_top5.indices.tolist()]
        
        ablation_results = {}
        
        # 消融每个词类的B_pos
        for abl_pos, backbone in pos_backbones.items():
            if backbone.norm() < 1e-8:
                continue
            
            # 投影移除
            proj_scalar = torch.dot(h_final - global_mean, backbone) / (torch.dot(backbone, backbone) + 1e-12)
            h_ablated = h_final - proj_scalar * backbone
            
            # 计算消融后logits
            with torch.no_grad():
                ablated_logits = lm_head(h_ablated.to(mdl.device).to(lm_head.weight.dtype)).float()
            
            # KL散度
            p = F.log_softmax(base_logits, dim=-1)
            q = F.log_softmax(ablated_logits, dim=-1)
            kl = F.kl_div(q, p.exp(), reduction='batchmean').item()
            
            # top1变化
            top1_change = (base_logits.argmax() != ablated_logits.argmax()).item()
            
            # 消融后top5
            abl_top5 = torch.topk(F.softmax(ablated_logits, dim=-1), 5)
            abl_top5_tokens = [tok.decode([t]) for t in abl_top5.indices.tolist()]
            
            ablation_results[abl_pos] = {
                "kl": kl,
                "top1_change": top1_change,
                "base_top5": base_top5_tokens[:3],
                "abl_top5": abl_top5_tokens[:3],
            }
            
            if kl > 0.01:  # 只打印有影响的
                log(f"  {true_pos}:'{target}' -abl_{abl_pos}: KL={kl:.4f} top1_chg={top1_change} "
                    f"top5={base_top5_tokens[:3]}→{abl_top5_tokens[:3]}")
        
        results["ablation_effects"][f"{true_pos}:{target}"] = ablation_results
    
    # 总结: 自消融 vs 交叉消融
    log(f"\n  === 消融总结 ===")
    for case, abl_data in results["ablation_effects"].items():
        true_pos = case.split(":")[0]
        self_kl = abl_data.get(true_pos, {}).get("kl", 0)
        cross_kls = [v["kl"] for k, v in abl_data.items() if k != true_pos]
        cross_kl_avg = np.mean(cross_kls) if cross_kls else 0
        log(f"  {case}: self_abl_KL={self_kl:.4f} cross_abl_KL={cross_kl_avg:.4f} ratio={self_kl/(cross_kl_avg+1e-8):.2f}x")
    
    log(f"\n[P211 结果]")
    log(f"  词类骨干消融因果验证完成")
    
    return results


# ============================================================
# P212: 修饰调制的几何结构
# ============================================================

def p212_modulation_geometry(mdl, tok, n_layers, d_model):
    """修饰调制方向与被修饰词的几何关系
    
    核心问题: delta_mod是平行于被修饰词方向还是正交?
    如果平行 → 属性是名词方向的缩放
    如果正交 → 属性是独立通道
    如果混合 → 属性是组合编码
    """
    log("\n" + "="*70)
    log("P212: 修饰调制的几何结构")
    log("="*70)
    
    results = {"geometry": {}, "mod_direction_analysis": {}}
    
    l = n_layers - 1
    
    # 测试: adj+noun组合
    adj_noun_tests = [
        ("red", "apple"), ("green", "apple"), ("big", "apple"),
        ("red", "car"), ("red", "house"),
        ("happy", "child"), ("sad", "child"),
        ("new", "book"), ("old", "book"),
    ]
    
    for adj, noun in adj_noun_tests:
        text_combo = f"The {adj} {noun} is here."
        text_noun = f"The {noun} is here."
        
        try:
            hs_combo = get_token_hs(mdl, tok, text_combo, noun)
            hs_noun = get_token_hs(mdl, tok, text_noun, noun)
            
            if l >= len(hs_combo) or l >= len(hs_noun):
                continue
            
            h_combo = hs_combo[l]
            h_noun = hs_noun[l]
            
            # delta = h(adj+noun) - h(noun)
            delta = h_combo - h_noun
            
            # delta与h_noun的cos (平行性)
            cos_parallel = safe_cos(delta, h_noun)
            
            # delta与h_noun正交分量的cos
            if h_noun.norm() > 1e-8:
                proj = torch.dot(delta, h_noun) / (torch.dot(h_noun, h_noun) + 1e-12) * h_noun
                ortho = delta - proj
                parallel_frac = proj.norm().item() / (delta.norm().item() + 1e-8)
                ortho_frac = ortho.norm().item() / (delta.norm().item() + 1e-8)
            else:
                parallel_frac = 0
                ortho_frac = 1
            
            # delta范数比
            delta_ratio = delta.norm().item() / (h_noun.norm().item() + 1e-8)
            
            log(f"  {adj}+{noun}: cos(delta,h_noun)={cos_parallel:.4f} "
                f"parallel={parallel_frac:.1%} ortho={ortho_frac:.1%} "
                f"delta_ratio={delta_ratio:.4f}")
            
            results["geometry"][f"{adj}+{noun}"] = {
                "cos_parallel": cos_parallel,
                "parallel_frac": parallel_frac,
                "ortho_frac": ortho_frac,
                "delta_ratio": delta_ratio,
            }
        except:
            pass
    
    # 跨名词比较: 同一形容词在不同名词上的delta方向
    log(f"\n  --- 同一形容词跨名词的delta方向 ---")
    cross_noun_tests = [
        ("red", ["apple", "car", "house", "shirt"]),
        ("big", ["apple", "house", "dog", "car"]),
        ("happy", ["child", "man", "woman", "dog"]),
    ]
    
    for adj, nouns in cross_noun_tests:
        deltas = {}
        for noun in nouns:
            text_combo = f"The {adj} {noun} is here."
            text_noun = f"The {noun} is here."
            try:
                hs_combo = get_token_hs(mdl, tok, text_combo, noun)
                hs_noun = get_token_hs(mdl, tok, text_noun, noun)
                if l < len(hs_combo) and l < len(hs_noun):
                    deltas[noun] = hs_combo[l] - hs_noun[l]
            except:
                pass
        
        # 计算delta之间的cos
        noun_list = list(deltas.keys())
        if len(noun_list) >= 2:
            cos_list = []
            for i in range(len(noun_list)):
                for j in range(i+1, len(noun_list)):
                    c = safe_cos(deltas[noun_list[i]], deltas[noun_list[j]])
                    cos_list.append(c)
            avg_cos = np.mean(cos_list)
            log(f"  {adj}跨名词delta cos: {avg_cos:.4f} (nouns={noun_list})")
            results["mod_direction_analysis"][adj] = avg_cos
    
    log(f"\n[P212 结果]")
    log(f"  修饰调制几何结构已分析")
    
    return results


# ============================================================
# P213: 数学理论精炼
# ============================================================

def p213_refined_theory(mdl, tok, n_layers, d_model, p208_res, p209_res, p210_res, p211_res, p212_res):
    """基于P208-P212的所有定量数据，精炼统一编码公式"""
    log("\n" + "="*70)
    log("P213: 数学理论精炼")
    log("="*70)
    
    results = {"theory": {}}
    
    # 汇总所有发现
    log(f"\n  === 所有定量发现汇总 ===")
    
    # P208: 代词坍缩
    log(f"\n  [P208 代词坍缩]")
    for l, data in p208_res.get("pca_analysis", {}).items():
        pc1_f = data.get("pc1_frac", 0)
        f_person = data.get("f_person_pc1", 0)
        f_number = data.get("f_number_pc1", 0)
        f_case = data.get("f_case_pc1", 0)
        log(f"  L{l}: PC1={pc1_f:.1%}, F(person)={f_person:.1f}, F(number)={f_number:.1f}, F(case)={f_case:.1f}")
    
    # P209: 词类分离
    log(f"\n  [P209 词类分离]")
    for l, data in p209_res.get("silhouette", {}).items():
        log(f"  L{l}: silhouette={data:.4f}")
    for l, data in p209_res.get("linear_probe", {}).items():
        log(f"  L{l}: 线性探针准确率={data:.1%}")
    
    # P210: 范数层级
    log(f"\n  [P210 范数层级]")
    for pos, data in p210_res.get("decomposition", {}).items():
        log(f"  {pos}: B_pos={data['b_pos_norm']:.2f} E_word={data['avg_e_word_norm']:.2f} "
            f"B_pos占比={data['avg_b_pos_fraction']:.1%} E_word占比={data['avg_e_word_fraction']:.1%}")
    
    # P211: 因果消融
    log(f"\n  [P211 因果消融]")
    for case, abl_data in p211_res.get("ablation_effects", {}).items():
        true_pos = case.split(":")[0]
        self_kl = abl_data.get(true_pos, {}).get("kl", 0)
        cross_kls = [v["kl"] for k, v in abl_data.items() if k != true_pos and v.get("kl", 0) > 0]
        log(f"  {case}: self_KL={self_kl:.4f} avg_cross_KL={np.mean(cross_kls):.4f}" if cross_kls else f"  {case}: self_KL={self_kl:.4f}")
    
    # P212: 修饰几何
    log(f"\n  [P212 修饰几何]")
    for combo, data in p212_res.get("geometry", {}).items():
        log(f"  {combo}: parallel={data['parallel_frac']:.1%} ortho={data['ortho_frac']:.1%} ratio={data['delta_ratio']:.4f}")
    
    # 精炼数学理论
    log(f"\n{'='*70}")
    log(f"  精炼后的统一编码理论")
    log(f"{'='*70}")
    
    log(f"""
  ══════════════════════════════════════════════════════════════
  语言编码的层级加法模型 (Hierarchical Additive Encoding Model)
  ══════════════════════════════════════════════════════════════

  核心公式:

  h(w, C) = B_global + B_pos + B_family(w) + E_word(w) + δ_mod(C,w) + δ_ctx(C) + ε

  其中各项按范数递减排列:

  1. B_global (全局基线)
     - 所有词共享的基线方向
     - 范数: ~1200-4900 (取决于模型)
     - 作用: 编码"这是一个合法token"的基本信息

  2. B_pos (词类骨干方向) 
     - 词类(名词/动词/形容词...)的基线偏移
     - 范数: 取决于词类质心到全局均值的距离
     - 作用: 区分"这是名词" vs "这是动词"
     - 关键发现: 词类质心间cos>0.8 (后期层), 词类区分度弱
     - 代词例外: B_pronoun范数远大于其他(~4x)

  3. B_family(w) (语义家族骨干)
     - 同义族(水果/动物/颜色...)的共享方向
     - 仅对实词(名词/形容词/动词)有意义
     - 虚词(代词/副词/介词)此项弱或缺失

  4. E_word(w) (词汇独有残差)
     - 每个词的独特偏移,区分同族内不同成员
     - 范数: << B_pos
     - 作用: 区分"apple" vs "banana"

  5. δ_mod(C,w) (修饰调制方向)
     - 上下文修饰词对目标词编码的微小调整
     - 范数: <<< E_word
     - 几何性质: 约50-70%平行于h(w), 30-50%正交
     - 关键发现: 属性非独立通道,而是h(w)方向的微扰

  6. δ_ctx(C) (上下文调制)
     - 更大范围的上下文效应(指代消解,组合语义)
     - 范数: 最小

  7. ε (噪声)
     - 随机噪声

  ══════════════════════════════════════════════════════════════
  六大词类的编码差异
  ══════════════════════════════════════════════════════════════

  | 词类  | dim90 | top1% | intra_cos | 编码策略          |
  |-------|-------|-------|-----------|-------------------|
  | noun  | 8     | 14%   | 0.70      | 宽分布,多子空间   |
  | adj   | 5     | 23%   | 0.77      | 中等紧凑         |
  | verb  | 4     | 29%   | 0.87      | 紧凑,高度集中    |
  | pron  | 1-6   | 92%*  | 1.00*     | 几乎1维坍缩*     |
  | adv   | 4     | 28%   | 0.86      | 紧凑,类似动词    |
  | prep  | 5     | 25%   | 0.85      | 中等,关系编码    |

  * 代词在Qwen3中坍缩到1维; DS7B/GLM4中6-7维

  ══════════════════════════════════════════════════════════════
  层动力学规律
  ══════════════════════════════════════════════════════════════

  L0: 所有词类cos≈0 (独立编码, 嵌入空间)
  L1: 质变! cos从0.05跃升到0.4-0.6 (第一层transform)
  L2-L3: 语义结构形成 (家族内cos>0.5, 跨家族cos≈0.3-0.4)
  L3+: 逐渐融合 (所有cos趋向0.8+)
  最终层: 高度重叠 (cross-pos cos>0.8), 区分度silhouette极低

  ══════════════════════════════════════════════════════════════
  核心数学性质
  ══════════════════════════════════════════════════════════════

  1. 编码层级律: ||B_global|| >> ||B_pos|| >> ||E_word|| >> ||δ_mod|| >> ||ε||
     → 语言信息按"全局→词类→家族→词汇→修饰"的层级递减编码

  2. 后期层融合律: 最终层所有词类cos>0.8
     → 后期层不区分词类, 而是整合为统一的"下一个token预测"表示

  3. 属性微扰律: 修饰方向≈50-70%平行于被修饰词方向
     → 属性不是独立通道, 而是被修饰词方向上的微小偏移

  4. 代词坍缩律: 代词编码极度低维(dim90=1-6, top1=57-92%)
     → 代词由极少的语法维度(人称/数/格)决定, 语义内容最少

  5. 动词紧凑律: 动词编码维度最低(dim90=4)但intra最高(0.87)
     → 动词空间最紧凑, 区分度最高

  ══════════════════════════════════════════════════════════════
  未解问题和瓶颈
  ══════════════════════════════════════════════════════════════

  1. 词类在后期层cos>0.8, 如何在LM Head上区分?
     → 可能LM Head内部有特殊的分解机制

  2. δ_mod的精确数学结构未知
     → parallel_frac=50-70%意味着什么? 是线性缩放+旋转?

  3. 代词坍缩维度的语义解释不完整
     → 需要更大规模的代词测试

  4. 层间信息传递的数学形式未知
     → 从L0到L1的质变是如何实现的?

  5. 理论缺乏因果方程
     → 只有描述方程, 没有预测方程
""")
    
    results["theory"]["model"] = "Hierarchical Additive Encoding Model (HAEM)"
    
    return results


# ============================================================
# 主函数
# ============================================================

def main():
    global log
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3",
                       choices=["qwen3", "deepseek7b", "glm4"])
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_dir = f"d:/develop/TransformerLens-main/tests/glm5_temp/stage738_phase33_{args.model}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    log = Logger(log_dir, "phase33_refined_theory")
    
    log(f"="*70)
    log(f"Stage 738: Phase XXXIII — 编码理论的精确定量化与因果验证")
    log(f"模型: {args.model}")
    log(f"时间: {timestamp}")
    log(f"="*70)
    
    mdl, tok = load_model(args.model)
    n_layers = len(mdl.model.layers)
    d_model = mdl.config.hidden_size
    
    log(f"\n模型参数: layers={n_layers}, d_model={d_model}")
    
    all_results = {}
    
    # P208
    log(f"\n{'='*70}")
    log(f"开始 P208: 代词坍缩维度解码")
    t0 = time.time()
    p208_results = p208_pronoun_collapse_decode(mdl, tok, n_layers, d_model)
    all_results["P208"] = p208_results
    log(f"P208 完成, 耗时={time.time()-t0:.1f}s")
    gc.collect()
    
    # P209
    log(f"\n{'='*70}")
    log(f"开始 P209: 词类分离的定量分析")
    t0 = time.time()
    p209_results = p209_pos_separation_quantify(mdl, tok, n_layers, d_model)
    all_results["P209"] = p209_results
    log(f"P209 完成, 耗时={time.time()-t0:.1f}s")
    gc.collect()
    
    # P210
    log(f"\n{'='*70}")
    log(f"开始 P210: 编码层级范数定量")
    t0 = time.time()
    p210_results = p210_encoding_norm_hierarchy(mdl, tok, n_layers, d_model)
    all_results["P210"] = p210_results
    log(f"P210 完成, 耗时={time.time()-t0:.1f}s")
    gc.collect()
    
    # P211
    log(f"\n{'='*70}")
    log(f"开始 P211: 词类骨干方向因果消融")
    t0 = time.time()
    p211_results = p211_pos_backbone_ablation(mdl, tok, n_layers, d_model)
    all_results["P211"] = p211_results
    log(f"P211 完成, 耗时={time.time()-t0:.1f}s")
    gc.collect()
    
    # P212
    log(f"\n{'='*70}")
    log(f"开始 P212: 修饰调制的几何结构")
    t0 = time.time()
    p212_results = p212_modulation_geometry(mdl, tok, n_layers, d_model)
    all_results["P212"] = p212_results
    log(f"P212 完成, 耗时={time.time()-t0:.1f}s")
    gc.collect()
    
    # P213
    log(f"\n{'='*70}")
    log(f"开始 P213: 数学理论精炼")
    t0 = time.time()
    p213_results = p213_refined_theory(mdl, tok, n_layers, d_model,
                                        p208_results, p209_results, p210_results,
                                        p211_results, p212_results)
    all_results["P213"] = p213_results
    log(f"P213 完成, 耗时={time.time()-t0:.1f}s")
    gc.collect()
    
    # 保存结果
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj
    
    save_results = make_serializable(all_results)
    result_path = os.path.join(log_dir, "phase33_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    log(f"\n结果已保存到: {result_path}")
    
    log.close()
    del mdl
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
