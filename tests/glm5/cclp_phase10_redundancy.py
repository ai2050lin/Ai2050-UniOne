"""
CCL-P(Phase 10): 冗余编码深入验证 — 扩展语法角色 + 低维流形 + 语义vs语法
=============================================================================
核心问题:
  Phase 9发现: 语法角色信息在W_U所有频段冗余编码, 每个频段都能100%分类4个角色
  但:
  1. 4个角色太少了! 10+个角色时冗余编码还成立吗?
  2. 4个角色只需3维分类, K个角色需要多少维? 语法内在维度是多少?
  3. 语义信息和语法信息在频段中的分布是否不同?
  4. 冗余编码的抗噪声能力如何?

实验:
  Exp1: ★★★★★ 扩展语法角色集(10+角色)的冗余编码
    → 添加: poss(所有格), det(限定词), aux(助动词), mark(标记词),
             case(格标记), nummod(数词修饰), compound(复合词), cc(并列连词)
    → 测试: 每个频段还能100%分类吗?

  Exp2: ★★★★★ 语法角色的低维流形 — 内在维度
    → 对K=4,6,8,10,12个角色, 测量分类所需最小维度
    → PCA降维: 从d_model逐步降到1维, 记录准确率
    → 语法内在维度 = 分类准确率达到95%所需的最小维度

  Exp3: ★★★★★ 语义vs语法的频段分布差异
    → 同一个词在不同语法角色中的hidden states (语法维度)
    → 不同语义类别(人/动物/物体/抽象)的hidden states (语义维度)
    → 语法信息 vs 语义信息 在W_U频段中的分布差异

  Exp4: ★★★★★ 冗余编码的抗噪声能力
    → 在不同频段添加高斯噪声, 测量分类准确率下降
    → 冗余编码: 单频段噪声不应完全破坏分类
    → 对比: 如果只在某一频段编码, 该频段噪声会完全破坏分类
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import argparse
import json
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from model_utils import load_model, get_layers, get_model_info, release_model, safe_decode, get_W_U


# ===== 扩展数据集: 10个语法角色 =====
# 每个角色至少12个样本, 使用自然英文句子
EXTENDED_ROLES_DATA = {
    "nsubj": {
        "desc": "主语名词",
        "sentences": [
            "The king ruled the kingdom", "The doctor treated the patient",
            "The artist painted the portrait", "The soldier defended the castle",
            "The cat sat on the mat", "The dog ran through the park",
            "The woman drove the car", "The man fixed the roof",
            "The student read the textbook", "The teacher explained the lesson",
            "The president signed the bill", "The chef cooked the meal",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "cat", "dog",
            "woman", "man", "student", "teacher", "president", "chef",
        ],
    },
    "dobj": {
        "desc": "直接宾语",
        "sentences": [
            "They crowned the king yesterday", "She visited the doctor recently",
            "He admired the artist greatly", "We honored the soldier today",
            "She chased the cat away", "He found the dog outside",
            "The police arrested the man quickly", "The company hired the woman recently",
            "I praised the student loudly", "You thanked the teacher warmly",
            "The nation elected the president fairly", "The customer tipped the chef generously",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "cat", "dog",
            "man", "woman", "student", "teacher", "president", "chef",
        ],
    },
    "amod": {
        "desc": "形容词修饰语",
        "sentences": [
            "The brave king fought hard", "The kind doctor helped many",
            "The creative artist worked well", "The strong soldier marched far",
            "The beautiful cat sat quietly", "The large dog ran swiftly",
            "The old woman walked slowly", "The tall man stood quietly",
            "The bright student read carefully", "The wise teacher explained clearly",
            "The powerful president decided firmly", "The skilled chef cooked perfectly",
        ],
        "target_words": [
            "brave", "kind", "creative", "strong", "beautiful", "large",
            "old", "tall", "bright", "wise", "powerful", "skilled",
        ],
    },
    "advmod": {
        "desc": "副词修饰语",
        "sentences": [
            "The king ruled wisely forever", "The doctor worked carefully always",
            "The artist painted beautifully daily", "The soldier fought bravely there",
            "The cat ran quickly home", "The dog barked loudly today",
            "The woman drove slowly home", "The man spoke quietly now",
            "The student read carefully alone", "The teacher spoke clearly again",
            "The president spoke firmly today", "The chef worked quickly then",
        ],
        "target_words": [
            "wisely", "carefully", "beautifully", "bravely", "quickly", "loudly",
            "slowly", "quietly", "carefully", "clearly", "firmly", "quickly",
        ],
    },
    "poss": {
        "desc": "所有格修饰语",
        "sentences": [
            "The king's crown glittered brightly", "The doctor's office opened early",
            "The artist's studio looked beautiful", "The soldier's uniform was clean",
            "The cat's tail swished gently", "The dog's bark echoed loudly",
            "The woman's dress looked elegant", "The man's car drove fast",
            "The student's essay read well", "The teacher's book sold quickly",
            "The president's speech inspired many", "The chef's restaurant opened today",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "cat", "dog",
            "woman", "man", "student", "teacher", "president", "chef",
        ],
    },
    "det": {
        "desc": "限定词(冠词/指示词)",
        "sentences": [
            "This king ruled very wisely", "That doctor helped many people",
            "This artist created beautiful works", "That soldier fought very bravely",
            "This cat jumped over the fence", "That dog chased the ball",
            "This woman led the team well", "That man built the house",
            "This student solved the problem", "That teacher taught the class",
            "This president changed the law", "That chef prepared the feast",
        ],
        "target_words": [
            "This", "That", "This", "That", "This", "That",
            "This", "That", "This", "That", "This", "That",
        ],
    },
    "aux": {
        "desc": "助动词",
        "sentences": [
            "The king has ruled for years", "The doctor will treat the patient",
            "The artist can paint very well", "The soldier should defend the fort",
            "The cat could jump very high", "The dog must run every day",
            "The woman has driven to work", "The man will fix the car",
            "The student should read the book", "The teacher can explain clearly",
            "The president must decide today", "The chef has cooked the meal",
        ],
        "target_words": [
            "has", "will", "can", "should", "could", "must",
            "has", "will", "should", "can", "must", "has",
        ],
    },
    "mark": {
        "desc": "从句标记词",
        "sentences": [
            "He ruled because the king commanded", "She studied since the doctor advised",
            "They painted while the artist watched", "We marched although the soldier hesitated",
            "It slept because the cat was tired", "He barked while the dog played",
            "She worked because the woman needed money", "He rested since the man was exhausted",
            "They studied because the student wanted grades", "She taught while the teacher observed",
            "He decided because the president ordered", "She cooked while the chef supervised",
        ],
        "target_words": [
            "because", "since", "while", "although", "because", "while",
            "because", "since", "because", "while", "because", "while",
        ],
    },
    "nummod": {
        "desc": "数词修饰语",
        "sentences": [
            "Three kings ruled the land", "Five doctors worked at the hospital",
            "Two artists painted the mural", "Four soldiers guarded the gate",
            "Seven cats sat on the wall", "Three dogs ran in the park",
            "Two women drove to the city", "Five men stood in the line",
            "Three students passed the exam", "Four teachers attended the meeting",
            "Two presidents signed the treaty", "Six chefs prepared the banquet",
        ],
        "target_words": [
            "Three", "Five", "Two", "Four", "Seven", "Three",
            "Two", "Five", "Three", "Four", "Two", "Six",
        ],
    },
    "cc": {
        "desc": "并列连词",
        "sentences": [
            "The king and the queen ruled together", "The doctor or the nurse helped first",
            "The artist and the musician performed", "The soldier but not the civilian fought",
            "The cat and the dog played together", "The dog or the cat chased the mouse",
            "The woman and the man worked together", "The teacher or the student answered first",
            "The student and the researcher collaborated", "The chef and the waiter served dinner",
            "The president and the minister agreed", "The doctor but not the patient decided",
        ],
        "target_words": [
            "and", "or", "and", "but", "and", "or",
            "and", "or", "and", "and", "and", "but",
        ],
    },
}

# 选择用于不同实验的角色组合
ROLE_SETS = {
    "4roles": ["nsubj", "dobj", "amod", "advmod"],
    "6roles": ["nsubj", "dobj", "amod", "advmod", "poss", "aux"],
    "8roles": ["nsubj", "dobj", "amod", "advmod", "poss", "aux", "det", "mark"],
    "10roles": list(EXTENDED_ROLES_DATA.keys()),
}

# 语义类别数据
SEMANTIC_DATA = {
    "person": {
        "words": ["king", "doctor", "artist", "soldier", "woman", "man", "student", "teacher", "president", "chef"],
        "sentences": [
            "The king ruled wisely", "The doctor treated patients",
            "The artist painted well", "The soldier fought bravely",
            "The woman worked hard", "The man walked slowly",
            "The student read carefully", "The teacher explained clearly",
            "The president decided firmly", "The chef cooked perfectly",
        ],
    },
    "animal": {
        "words": ["cat", "dog", "bird", "fish", "horse", "rabbit", "tiger", "eagle", "dolphin", "snake"],
        "sentences": [
            "The cat jumped high", "The dog barked loudly",
            "The bird sang beautifully", "The fish swam quickly",
            "The horse ran fast", "The rabbit hopped away",
            "The tiger roared fiercely", "The eagle soared high",
            "The dolphin jumped gracefully", "The snake slithered quietly",
        ],
    },
    "object": {
        "words": ["table", "chair", "house", "car", "book", "phone", "computer", "door", "window", "bridge"],
        "sentences": [
            "The table stood firmly", "The chair broke suddenly",
            "The house looked old", "The car drove fast",
            "The book sold well", "The phone rang loudly",
            "The computer worked perfectly", "The door opened slowly",
            "The window shattered completely", "The bridge collapsed unexpectedly",
        ],
    },
    "abstract": {
        "words": ["freedom", "justice", "love", "truth", "beauty", "wisdom", "courage", "peace", "hope", "fear"],
        "sentences": [
            "Freedom mattered deeply", "Justice prevailed finally",
            "Love endured forever", "Truth emerged gradually",
            "Beauty inspired greatly", "Wisdom guided wisely",
            "Courage sustained many", "Peace spread slowly",
            "Hope remained strong", "Fear gripped tightly",
        ],
    },
}


def find_token_index(tokens, word):
    word_lower = word.lower()
    word_start = word_lower[:3]
    for i, tok in enumerate(tokens):
        tok_lower = tok.lower().strip()
        if word_lower in tok_lower or tok_lower.startswith(word_start):
            return i
    # fallback: 搜索包含目标词的token
    for i, tok in enumerate(tokens):
        if word_lower[:2] in tok.lower():
            return i
    return None


def collect_hidden_states_multirole(model, tokenizer, device, role_names, data_dict, layer_idx=-1):
    """收集多个语法角色的hidden states"""
    all_h = []
    all_labels = []
    
    layers = get_layers(model)
    target_layer = layers[layer_idx] if layer_idx >= 0 else layers[-1]
    
    for role_idx, role in enumerate(role_names):
        data = data_dict[role]
        for sent, target_word in zip(data["sentences"], data["target_words"]):
            toks = tokenizer(sent, return_tensors="pt").to(device)
            input_ids = toks.input_ids
            tokens_list = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]
            dep_idx = find_token_index(tokens_list, target_word)
            if dep_idx is None:
                continue
            
            captured = {}
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    captured['h'] = output[0].detach().float().cpu().numpy()
                else:
                    captured['h'] = output.detach().float().cpu().numpy()
            
            h_handle = target_layer.register_forward_hook(hook_fn)
            with torch.no_grad():
                _ = model(**toks)
            h_handle.remove()
            
            if 'h' not in captured:
                continue
            
            h_vec = captured['h'][0, dep_idx, :]
            all_h.append(h_vec)
            all_labels.append(role_idx)
    
    return np.array(all_h), np.array(all_labels)


def collect_semantic_hidden_states(model, tokenizer, device, layer_idx=-1):
    """收集语义类别的hidden states"""
    all_h = []
    all_labels = []
    
    layers = get_layers(model)
    target_layer = layers[layer_idx] if layer_idx >= 0 else layers[-1]
    
    for cat_idx, (cat_name, cat_data) in enumerate(SEMANTIC_DATA.items()):
        for sent in cat_data["sentences"]:
            toks = tokenizer(sent, return_tensors="pt").to(device)
            
            captured = {}
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    captured['h'] = output[0].detach().float().cpu().numpy()
                else:
                    captured['h'] = output.detach().float().cpu().numpy()
            
            h_handle = target_layer.register_forward_hook(hook_fn)
            with torch.no_grad():
                _ = model(**toks)
            h_handle.remove()
            
            if 'h' not in captured:
                continue
            
            # 取最后一个实词token
            h_vec = captured['h'][0, -1, :]
            all_h.append(h_vec)
            all_labels.append(cat_idx)
    
    return np.array(all_h), np.array(all_labels)


def compute_W_U_eigenspectrum(W_U):
    """计算W_U的完整特征值谱和特征向量"""
    d_model = W_U.shape[1]
    W_U_f64 = W_U.astype(np.float64)
    
    print(f"  计算W_U^T @ W_U ({d_model}x{d_model})...")
    WtW = W_U_f64.T @ W_U_f64
    
    eigenvalues, eigenvectors = np.linalg.eigh(WtW)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    
    return eigenvalues, eigenvectors


# ===== Exp1: 扩展语法角色集的冗余编码 =====
def exp1_extended_roles_redundancy(model, tokenizer, device):
    """测试10个语法角色的冗余编码"""
    print("\n" + "="*70)
    print("Exp1: 扩展语法角色集(10角色)的冗余编码 ★★★★★")
    print("="*70)
    
    results = {}
    
    for set_name, role_names in ROLE_SETS.items():
        n_roles = len(role_names)
        print(f"\n  --- {set_name} ({n_roles}角色) ---")
        
        # 收集hidden states
        print(f"  收集hidden states...")
        H, labels = collect_hidden_states_multirole(
            model, tokenizer, device, role_names, EXTENDED_ROLES_DATA)
        print(f"  收集到 {len(H)} 个样本")
        
        if len(H) < n_roles * 2:
            print(f"  样本不足, 跳过")
            continue
        
        # 全空间分类
        scaler = StandardScaler()
        H_scaled = scaler.fit_transform(H)
        probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
        cv_full = cross_val_score(probe, H_scaled, labels, cv=5, scoring='accuracy')
        print(f"  全空间分类: CV={cv_full.mean():.4f} ± {cv_full.std():.4f}")
        
        # 随机基线
        random_acc = 1.0 / n_roles
        print(f"  随机基线: {random_acc:.4f}")
        
        # W_U频段分析
        W_U = get_W_U(model)
        eigenvalues, eigenvectors = compute_W_U_eigenspectrum(W_U)
        d_model = H.shape[1]
        
        # 6个频段
        bands = {
            'top10': (0, d_model // 10),
            '10-25': (d_model // 10, d_model // 4),
            '25-50': (d_model // 4, d_model // 2),
            '50-75': (d_model // 2, 3 * d_model // 4),
            '75-90': (3 * d_model // 4, 9 * d_model // 10),
            'bottom10': (9 * d_model // 10, d_model),
        }
        
        band_cvs = {}
        print(f"\n  频段分类准确率:")
        print(f"  {'Band':>10s} {'Dim':>5s} {'CV':>8s} {'vs_random':>10s}")
        
        for band_name, (start, end) in bands.items():
            dim = end - start
            U_band = eigenvectors[:, start:end]
            H_band = H @ U_band
            
            probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
            try:
                cv = cross_val_score(probe, H_band, labels, cv=5, scoring='accuracy')
                cv_mean = cv.mean()
            except Exception:
                cv_mean = 0.0
            
            vs_random = cv_mean / random_acc
            print(f"  {band_name:>10s} {dim:5d} {cv_mean:8.4f} {vs_random:10.1f}x")
            band_cvs[band_name] = float(cv_mean)
        
        # 高能vs低能50%
        U_high = eigenvectors[:, :d_model//2]
        U_low = eigenvectors[:, d_model//2:]
        H_high = H @ U_high
        H_low = H @ U_low
        
        probe_h = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
        probe_l = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
        
        try:
            cv_high = cross_val_score(probe_h, H_high, labels, cv=5, scoring='accuracy')
            cv_low = cross_val_score(probe_l, H_low, labels, cv=5, scoring='accuracy')
        except Exception:
            cv_high = np.array([0.0])
            cv_low = np.array([0.0])
        
        print(f"\n  ★ 高能50%: CV={cv_high.mean():.4f} ({cv_high.mean()/random_acc:.1f}x random)")
        print(f"  ★ 低能50%: CV={cv_low.mean():.4f} ({cv_low.mean()/random_acc:.1f}x random)")
        print(f"  ★ low/high比: {cv_low.mean()/max(cv_high.mean(),0.001):.2f}")
        
        results[set_name] = {
            'n_roles': n_roles,
            'n_samples': len(H),
            'full_cv': float(cv_full.mean()),
            'random_baseline': float(random_acc),
            'band_cvs': band_cvs,
            'high50_cv': float(cv_high.mean()),
            'low50_cv': float(cv_low.mean()),
            'low_high_ratio': float(cv_low.mean() / max(cv_high.mean(), 0.001)),
        }
    
    # ★ 关键总结
    print(f"\n{'='*70}")
    print(f"  ★ Exp1 总结: 冗余编码 vs 角色数量")
    print(f"  {'Roles':>6s} {'Full_CV':>8s} {'High50':>8s} {'Low50':>8s} {'Ratio':>6s}")
    for set_name, r in results.items():
        print(f"  {r['n_roles']:6d} {r['full_cv']:8.4f} {r['high50_cv']:8.4f} "
              f"{r['low50_cv']:8.4f} {r['low_high_ratio']:6.2f}")
    
    return results


# ===== Exp2: 低维流形 — 语法内在维度 =====
def exp2_intrinsic_dimension(model, tokenizer, device):
    """测量分类K个语法角色所需的最小维度"""
    print("\n" + "="*70)
    print("Exp2: 语法角色的低维流形 — 内在维度 ★★★★★")
    print("="*70)
    
    results = {}
    
    for set_name, role_names in ROLE_SETS.items():
        n_roles = len(role_names)
        print(f"\n  --- {set_name} ({n_roles}角色) ---")
        
        # 收集hidden states
        H, labels = collect_hidden_states_multirole(
            model, tokenizer, device, role_names, EXTENDED_ROLES_DATA)
        print(f"  收集到 {len(H)} 个样本")
        
        if len(H) < n_roles * 2:
            continue
        
        # PCA降维曲线
        max_dim = min(H.shape[0], H.shape[1]) - 1  # 不能超过样本数或特征数
        dims_to_test = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 100]
        dims_to_test = [d for d in dims_to_test if d < max_dim]
        
        dim_cvs = {}
        print(f"\n  PCA降维分类准确率:")
        print(f"  {'Dim':>5s} {'CV':>8s} {'vs_full':>8s}")
        
        for dim in dims_to_test:
            pca = PCA(n_components=dim)
            H_pca = pca.fit_transform(H)
            
            probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
            try:
                cv = cross_val_score(probe, H_pca, labels, cv=5, scoring='accuracy')
                cv_mean = cv.mean()
            except Exception:
                cv_mean = 0.0
            
            dim_cvs[dim] = float(cv_mean)
            print(f"  {dim:5d} {cv_mean:8.4f}")
        
        # 全空间准确率
        scaler = StandardScaler()
        H_scaled = scaler.fit_transform(H)
        probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
        cv_full = cross_val_score(probe, H_scaled, labels, cv=5, scoring='accuracy')
        full_cv = cv_full.mean()
        print(f"  Full {H.shape[1]:5d} {full_cv:8.4f}")
        
        # 找到95%和90%阈值维度
        dim_95 = None
        dim_90 = None
        threshold_95 = full_cv * 0.95
        threshold_90 = full_cv * 0.90
        
        for dim in sorted(dim_cvs.keys()):
            if dim_90 is None and dim_cvs[dim] >= threshold_90:
                dim_90 = dim
            if dim_95 is None and dim_cvs[dim] >= threshold_95:
                dim_95 = dim
        
        print(f"\n  内在维度(95%阈值): {dim_95}")
        print(f"  内在维度(90%阈值): {dim_90}")
        
        results[set_name] = {
            'n_roles': n_roles,
            'n_samples': len(H),
            'full_cv': float(full_cv),
            'dim_curve': dim_cvs,
            'intrinsic_dim_95': dim_95,
            'intrinsic_dim_90': dim_90,
        }
    
    # ★ 关键分析: 内在维度 vs 角色数量
    print(f"\n{'='*70}")
    print(f"  ★ Exp2 总结: 内在维度 vs 角色数量")
    print(f"  {'Roles':>6s} {'Full_CV':>8s} {'dim_95':>7s} {'dim_90':>7s}")
    for set_name, r in results.items():
        d95 = str(r['intrinsic_dim_95']) if r['intrinsic_dim_95'] is not None else 'N/A'
        d90 = str(r['intrinsic_dim_90']) if r['intrinsic_dim_90'] is not None else 'N/A'
        print(f"  {r['n_roles']:6d} {r['full_cv']:8.4f} "
              f"{d95:>7s} "
              f"{d90:>7s}")
    
    return results


# ===== Exp3: 语义vs语法的频段分布 =====
def exp3_semantic_vs_syntax_spectrum(model, tokenizer, device):
    """比较语义和语法信息在W_U频段中的分布差异"""
    print("\n" + "="*70)
    print("Exp3: 语义vs语法频段分布差异 ★★★★★")
    print("="*70)
    
    results = {}
    
    # ---- Part A: 语法角色分类(4角色) ----
    print(f"\n  Part A: 语法角色分类(4角色)")
    syntax_roles = ROLE_SETS["4roles"]
    H_syn, labels_syn = collect_hidden_states_multirole(
        model, tokenizer, device, syntax_roles, EXTENDED_ROLES_DATA)
    print(f"  语法: {len(H_syn)} 样本, {len(syntax_roles)} 角色")
    
    # ---- Part B: 语义类别分类(4类) ----
    print(f"\n  Part B: 语义类别分类(4类: 人/动物/物体/抽象)")
    H_sem, labels_sem = collect_semantic_hidden_states(model, tokenizer, device)
    print(f"  语义: {len(H_sem)} 样本, 4 类别")
    
    # W_U特征谱
    W_U = get_W_U(model)
    eigenvalues, eigenvectors = compute_W_U_eigenspectrum(W_U)
    d_model = H_syn.shape[1]
    
    # 频段定义
    n_bins = 10
    bin_size = d_model // n_bins
    
    # ---- Part C: 逐频段分类 ----
    print(f"\n  Part C: 逐频段分类准确率对比")
    print(f"  {'Bin':>5s} {'λ_range':>20s} {'Syntax_CV':>10s} {'Semantic_CV':>12s} {'Ratio':>7s}")
    
    bin_results = []
    syntax_rand = 1.0 / len(syntax_roles)
    semantic_rand = 1.0 / 4
    
    for bi in range(n_bins):
        start = bi * bin_size
        end = min((bi + 1) * bin_size, d_model)
        U_bin = eigenvectors[:, start:end]
        
        # 语法分类
        H_syn_bin = H_syn @ U_bin
        probe_syn = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
        try:
            cv_syn = cross_val_score(probe_syn, H_syn_bin, labels_syn, cv=5, scoring='accuracy')
            syn_cv = cv_syn.mean()
        except Exception:
            syn_cv = 0.0
        
        # 语义分类
        H_sem_bin = H_sem @ U_bin
        probe_sem = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
        try:
            cv_sem = cross_val_score(probe_sem, H_sem_bin, labels_sem, cv=5, scoring='accuracy')
            sem_cv = cv_sem.mean()
        except Exception:
            sem_cv = 0.0
        
        ratio = syn_cv / max(sem_cv, 0.001)
        lambda_range = f"[{eigenvalues[start]:.0f}, {eigenvalues[end-1]:.0f}]"
        print(f"  {bi+1:5d} {lambda_range:>20s} {syn_cv:10.4f} {sem_cv:12.4f} {ratio:7.2f}")
        
        bin_results.append({
            'bin': bi + 1,
            'start': start,
            'end': end,
            'lambda_start': float(eigenvalues[start]),
            'lambda_end': float(eigenvalues[end-1]),
            'syntax_cv': float(syn_cv),
            'semantic_cv': float(sem_cv),
            'ratio': float(ratio),
        })
    
    # ---- Part D: 探针方向频谱投影 ----
    print(f"\n  Part D: 探针方向在W_U频段中的投影能量分布")
    
    # 语法探针方向
    scaler_syn = StandardScaler()
    H_syn_scaled = scaler_syn.fit_transform(H_syn)
    probe_syn_full = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
    probe_syn_full.fit(H_syn_scaled, labels_syn)
    
    # 语义探针方向
    scaler_sem = StandardScaler()
    H_sem_scaled = scaler_sem.fit_transform(H_sem)
    probe_sem_full = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
    probe_sem_full.fit(H_sem_scaled, labels_sem)
    
    # 分析: 语法探针方向在特征值频段中的投影
    print(f"\n  语法探针方向投影:")
    for ri, role in enumerate(syntax_roles):
        w = probe_syn_full.coef_[ri]
        direction = w / max(np.linalg.norm(w), 1e-10)
        proj = eigenvectors.T @ direction
        proj_energy = proj ** 2
        
        # 各频段能量比
        high50 = np.sum(proj_energy[:d_model//2]) / np.sum(proj_energy)
        low50 = np.sum(proj_energy[d_model//2:]) / np.sum(proj_energy)
        
        # 加权特征值
        weighted_eigen = np.sum(eigenvalues * proj_energy) / max(np.sum(proj_energy), 1e-10)
        ratio = weighted_eigen / np.median(eigenvalues)
        
        print(f"    {role:>8s}: high50%={high50:.3f}, low50%={low50:.3f}, "
              f"weighted_λ_ratio={ratio:.2f}")
    
    # 分析: 语义探针方向在特征值频段中的投影
    print(f"\n  语义探针方向投影:")
    sem_cats = list(SEMANTIC_DATA.keys())
    for ri, cat in enumerate(sem_cats):
        w = probe_sem_full.coef_[ri]
        direction = w / max(np.linalg.norm(w), 1e-10)
        proj = eigenvectors.T @ direction
        proj_energy = proj ** 2
        
        high50 = np.sum(proj_energy[:d_model//2]) / np.sum(proj_energy)
        low50 = np.sum(proj_energy[d_model//2:]) / np.sum(proj_energy)
        
        weighted_eigen = np.sum(eigenvalues * proj_energy) / max(np.sum(proj_energy), 1e-10)
        ratio = weighted_eigen / np.median(eigenvalues)
        
        print(f"    {cat:>8s}: high50%={high50:.3f}, low50%={low50:.3f}, "
              f"weighted_λ_ratio={ratio:.2f}")
    
    # ---- Part E: 质心位移方向的频谱分布 ----
    print(f"\n  Part E: 质心位移方向频谱分布对比")
    
    # 语法质心
    syn_centers = {}
    for ri, role in enumerate(syntax_roles):
        mask = labels_syn == ri
        syn_centers[role] = H_syn[mask].mean(axis=0)
    syn_overall = H_syn.mean(axis=0)
    
    syn_displacement_spectrum = {}
    for role in syntax_roles:
        d = syn_centers[role] - syn_overall
        d_hat = d / max(np.linalg.norm(d), 1e-10)
        proj = eigenvectors.T @ d_hat
        proj_energy = proj ** 2
        high50 = np.sum(proj_energy[:d_model//2]) / np.sum(proj_energy)
        syn_displacement_spectrum[role] = high50
    
    # 语义质心
    sem_centers = {}
    for ci, cat in enumerate(sem_cats):
        mask = labels_sem == ci
        sem_centers[cat] = H_sem[mask].mean(axis=0)
    sem_overall = H_sem.mean(axis=0)
    
    sem_displacement_spectrum = {}
    for cat in sem_cats:
        d = sem_centers[cat] - sem_overall
        d_hat = d / max(np.linalg.norm(d), 1e-10)
        proj = eigenvectors.T @ d_hat
        proj_energy = proj ** 2
        high50 = np.sum(proj_energy[:d_model//2]) / np.sum(proj_energy)
        sem_displacement_spectrum[cat] = high50
    
    syn_mean_high50 = np.mean(list(syn_displacement_spectrum.values()))
    sem_mean_high50 = np.mean(list(sem_displacement_spectrum.values()))
    
    print(f"  语法质心位移 high50%均值: {syn_mean_high50:.3f}")
    print(f"  语义质心位移 high50%均值: {sem_mean_high50:.3f}")
    print(f"  差异: {abs(syn_mean_high50 - sem_mean_high50):.3f} "
          f"({'语法更偏高频' if syn_mean_high50 > sem_mean_high50 else '语义更偏高频'})")
    
    results = {
        'bin_results': bin_results,
        'syntax_random_baseline': float(syntax_rand),
        'semantic_random_baseline': float(semantic_rand),
        'syntax_displacement_high50': syn_displacement_spectrum,
        'semantic_displacement_high50': sem_displacement_spectrum,
        'syntax_mean_high50': float(syn_mean_high50),
        'semantic_mean_high50': float(sem_mean_high50),
    }
    
    return results


# ===== Exp4: 冗余编码的抗噪声能力 =====
def exp4_noise_robustness(model, tokenizer, device):
    """测试冗余编码的抗噪声能力"""
    print("\n" + "="*70)
    print("Exp4: 冗余编码的抗噪声能力 ★★★★★")
    print("="*70)
    
    # 收集4角色hidden states
    syntax_roles = ROLE_SETS["4roles"]
    H, labels = collect_hidden_states_multirole(
        model, tokenizer, device, syntax_roles, EXTENDED_ROLES_DATA)
    print(f"  收集到 {len(H)} 个样本")
    
    d_model = H.shape[1]
    
    # W_U特征谱
    W_U = get_W_U(model)
    eigenvalues, eigenvectors = compute_W_U_eigenspectrum(W_U)
    
    # 噪声水平
    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    
    # 噪声方式:
    # 1. 各向同性噪声: 在所有方向等量加噪
    # 2. 高能方向噪声: 只在top10%方向加噪
    # 3. 低能方向噪声: 只在bottom10%方向加噪
    
    results = {}
    
    # 全空间分类(无噪声)
    scaler = StandardScaler()
    H_scaled = scaler.fit_transform(H)
    probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
    cv_clean = cross_val_score(probe, H_scaled, labels, cv=5, scoring='accuracy')
    print(f"  无噪声全空间CV: {cv_clean.mean():.4f}")
    
    # 各向同性噪声
    print(f"\n  各向同性噪声:")
    print(f"  {'Noise':>6s} {'CV':>8s} {'Drop':>8s}")
    iso_results = {}
    for noise_level in noise_levels:
        H_noisy = H + np.random.randn(*H.shape) * noise_level * np.std(H)
        scaler_n = StandardScaler()
        H_noisy_scaled = scaler_n.fit_transform(H_noisy)
        probe_n = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
        cv_n = cross_val_score(probe_n, H_noisy_scaled, labels, cv=5, scoring='accuracy')
        drop = cv_clean.mean() - cv_n.mean()
        print(f"  {noise_level:6.2f} {cv_n.mean():8.4f} {drop:8.4f}")
        iso_results[noise_level] = float(cv_n.mean())
    
    # 方向性噪声
    d10 = d_model // 10
    U_high10 = eigenvectors[:, :d10]
    U_low10 = eigenvectors[:, -d10:]
    
    # 只在高能方向加噪
    print(f"\n  高能方向(top10%)噪声:")
    print(f"  {'Noise':>6s} {'CV':>8s} {'Drop':>8s}")
    high_noise_results = {}
    for noise_level in noise_levels:
        # 只在高能方向加噪声
        noise_raw = np.random.randn(len(H), d10) * noise_level * np.std(H)
        noise_in_full = noise_raw @ U_high10.T  # 投影回全空间
        H_noisy = H + noise_in_full
        scaler_n = StandardScaler()
        H_noisy_scaled = scaler_n.fit_transform(H_noisy)
        probe_n = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
        cv_n = cross_val_score(probe_n, H_noisy_scaled, labels, cv=5, scoring='accuracy')
        drop = cv_clean.mean() - cv_n.mean()
        print(f"  {noise_level:6.2f} {cv_n.mean():8.4f} {drop:8.4f}")
        high_noise_results[noise_level] = float(cv_n.mean())
    
    # 只在低能方向加噪
    print(f"\n  低能方向(bottom10%)噪声:")
    print(f"  {'Noise':>6s} {'CV':>8s} {'Drop':>8s}")
    low_noise_results = {}
    for noise_level in noise_levels:
        noise_raw = np.random.randn(len(H), d10) * noise_level * np.std(H)
        noise_in_full = noise_raw @ U_low10.T
        H_noisy = H + noise_in_full
        scaler_n = StandardScaler()
        H_noisy_scaled = scaler_n.fit_transform(H_noisy)
        probe_n = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
        cv_n = cross_val_score(probe_n, H_noisy_scaled, labels, cv=5, scoring='accuracy')
        drop = cv_clean.mean() - cv_n.mean()
        print(f"  {noise_level:6.2f} {cv_n.mean():8.4f} {drop:8.4f}")
        low_noise_results[noise_level] = float(cv_n.mean())
    
    # ★ 关键对比: 破坏单频段信息
    print(f"\n  ★ 单频段信息破坏测试:")
    print(f"  {'Method':>20s} {'CV':>8s} {'Drop':>8s}")
    
    # 完全移除高能50%信息
    U_high50 = eigenvectors[:, :d_model//2]
    U_low50 = eigenvectors[:, d_model//2:]
    H_low_only = H @ U_low50 @ U_low50.T  # 只保留低能50%
    scaler_lo = StandardScaler()
    H_low_only_scaled = scaler_lo.fit_transform(H_low_only)
    probe_lo = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
    cv_lo = cross_val_score(probe_lo, H_low_only_scaled, labels, cv=5, scoring='accuracy')
    print(f"  {'移除高能50%':>20s} {cv_lo.mean():8.4f} {cv_clean.mean()-cv_lo.mean():8.4f}")
    
    # 完全移除低能50%信息
    H_high_only = H @ U_high50 @ U_high50.T  # 只保留高能50%
    scaler_ho = StandardScaler()
    H_high_only_scaled = scaler_ho.fit_transform(H_high_only)
    probe_ho = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
    cv_ho = cross_val_score(probe_ho, H_high_only_scaled, labels, cv=5, scoring='accuracy')
    print(f"  {'移除低能50%':>20s} {cv_ho.mean():8.4f} {cv_clean.mean()-cv_ho.mean():8.4f}")
    
    results = {
        'clean_cv': float(cv_clean.mean()),
        'isotropic_noise': iso_results,
        'high_direction_noise': high_noise_results,
        'low_direction_noise': low_noise_results,
        'remove_high50_cv': float(cv_lo.mean()),
        'remove_low50_cv': float(cv_ho.mean()),
    }
    
    return results


# ===== 主函数 =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, required=True,
                       choices=[1, 2, 3, 4])
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"CCL-P Phase10 冗余编码深入 | Model={args.model} | Exp={args.exp}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    print(f"  Model: {model_info.model_class}, Layers={model_info.n_layers}, "
          f"d_model={model_info.d_model}")
    
    try:
        if args.exp == 1:
            results = exp1_extended_roles_redundancy(model, tokenizer, device)
        elif args.exp == 2:
            results = exp2_intrinsic_dimension(model, tokenizer, device)
        elif args.exp == 3:
            results = exp3_semantic_vs_syntax_spectrum(model, tokenizer, device)
        elif args.exp == 4:
            results = exp4_noise_robustness(model, tokenizer, device)
        
        # 保存结果
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..', 'glm5_temp')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir,
                               f"cclp_exp{args.exp}_{args.model}_results.json")
        
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(convert(v) for v in obj)
            return obj
        
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(convert(results), f, indent=2, ensure_ascii=False)
        print(f"\n  结果已保存: {out_path}")
    
    finally:
        release_model(model)
        print(f"  模型已释放")


if __name__ == "__main__":
    main()
