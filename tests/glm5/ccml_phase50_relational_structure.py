"""
Phase 50: Relational Structure Recovery — 关系结构恢复
======================================================

Phase 49核心发现:
  - 角色身份(agent/patient)在last-token归一化h中线性不可读(0.0)
  - 但结构类型(及物/不及物)完美可读(1.0)
  → 角色信息可能不在单token表示中，而在跨token关系中

Phase 50核心假设:
  语言结构 = Token间关系模式，不是单Token属性

五个实验:
  50A: Token级角色探测 — 从agent/patient的token位置探测角色
  50B: 跨Token关系提取 — h_i - h_j是否编码语法关系
  50C: 注意力模式分析 — Attn矩阵是否编码角色绑定
  50D: 关系图恢复 — 从token关系构建语法树
  50E: 全局上下文操作 — Δz = f(z, context)验证
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import argparse
import torch
import numpy as np
import gc
import time
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.spatial.distance import cosine
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from model_utils import (load_model, get_layers, get_model_info, release_model,
                          safe_decode, collect_layer_outputs, get_W_U)


# ============================================================
# 辅助函数
# ============================================================
def get_all_token_hidden_states(model, tokenizer, sentence, n_layers, device):
    """
    收集句子所有token在所有层的hidden states
    返回: {layer_idx: tensor[seq_len, d_model]}
    同时返回tokenized的input_ids
    """
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    with torch.no_grad():
        embed = model.get_input_embeddings()(input_ids)
        position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
        outputs = collect_layer_outputs(model, embed, position_ids, n_layers)
    
    result = {}
    for li in range(n_layers):
        key = f"L{li}"
        if key in outputs:
            result[li] = outputs[key][0].numpy()  # [seq_len, d_model]
    
    del outputs, embed
    return result, input_ids[0].cpu().numpy()


def get_attention_weights(model, tokenizer, sentence, n_layers, device, target_layers=None):
    """
    收集指定层的注意力权重
    返回: {layer_idx: tensor[n_heads, seq_len, seq_len]}
    """
    if target_layers is None:
        target_layers = list(range(n_layers))
    
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    layers = get_layers(model)
    attn_weights = {}
    
    def make_attn_hook(li):
        def hook(module, input, output):
            # output通常包含(attn_output, attn_weights, ...)
            if len(output) > 1 and output[1] is not None:
                attn_weights[li] = output[1].detach().float().cpu()[0].numpy()
        return hook
    
    hooks = []
    for li in target_layers:
        layer = layers[li]
        # 尝试获取self_attn模块
        if hasattr(layer, 'self_attn'):
            attn_mod = layer.self_attn
        elif hasattr(layer, 'attention'):
            attn_mod = layer.attention
        else:
            continue
        hooks.append(attn_mod.register_forward_hook(make_attn_hook(li)))
    
    with torch.no_grad():
        try:
            embed = model.get_input_embeddings()(input_ids)
            position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
            _ = model(inputs_embeds=embed, position_ids=position_ids, output_attentions=False)
        except Exception as e:
            pass
    
    for h in hooks:
        h.remove()
    
    del embed
    return attn_weights, input_ids[0].cpu().numpy()


def find_token_position(token_ids, tokenizer, target_word):
    """找到目标词在token序列中的位置(可能是多token)"""
    decoded = [safe_decode(tokenizer, tid) for tid in token_ids]
    # 尝试直接匹配
    positions = []
    for i, d in enumerate(decoded):
        if target_word.lower() in d.lower().strip():
            positions.append(i)
    if positions:
        return positions
    # 尝试子串匹配
    for i, d in enumerate(decoded):
        if target_word.lower() in d.lower():
            positions.append(i)
    return positions


# ============================================================
# 句子集定义
# ============================================================
# 带标注的句子集: (sentence, agent_word, patient_word, verb_word, structure_type)
ANNOTATED_SENTENCES = [
    # 及物句 - 动物作agent
    ("The cat chases the mouse.", "cat", "mouse", "chases", "transitive"),
    ("The dog bites the bone.", "dog", "bone", "bites", "transitive"),
    ("The bird catches the fish.", "bird", "fish", "catches", "transitive"),
    ("The fox hunts the rabbit.", "fox", "rabbit", "hunts", "transitive"),
    ("The bear eats the honey.", "bear", "honey", "eats", "transitive"),
    ("The cat watches the bird.", "cat", "bird", "watches", "transitive"),
    ("The dog follows the cat.", "dog", "cat", "follows", "transitive"),
    ("The lion attacks the deer.", "lion", "deer", "attacks", "transitive"),
    ("The wolf chases the sheep.", "wolf", "sheep", "chases", "transitive"),
    ("The hawk catches the snake.", "hawk", "snake", "catches", "transitive"),
    
    # 及物句 - 人作agent
    ("The man reads the book.", "man", "book", "reads", "transitive"),
    ("The woman opens the door.", "woman", "door", "opens", "transitive"),
    ("The boy throws the ball.", "boy", "ball", "throws", "transitive"),
    ("The girl paints the picture.", "girl", "picture", "paints", "transitive"),
    ("The teacher writes the letter.", "teacher", "letter", "writes", "transitive"),
    ("The doctor helps the patient.", "doctor", "patient", "helps", "transitive"),
    ("The farmer grows the corn.", "farmer", "corn", "grows", "transitive"),
    ("The chef cooks the meal.", "chef", "meal", "cooks", "transitive"),
    ("The driver starts the car.", "driver", "car", "starts", "transitive"),
    ("The artist draws the map.", "artist", "map", "draws", "transitive"),
    
    # 被动句
    ("The mouse is chased by the cat.", "cat", "mouse", "chased", "passive"),
    ("The bone is bitten by the dog.", "dog", "bone", "bitten", "passive"),
    ("The fish is caught by the bird.", "bird", "fish", "caught", "passive"),
    ("The book is read by the man.", "man", "book", "read", "passive"),
    ("The door is opened by the woman.", "woman", "door", "opened", "passive"),
    ("The ball is thrown by the boy.", "boy", "ball", "thrown", "passive"),
    ("The letter is written by the teacher.", "teacher", "letter", "written", "passive"),
    ("The rabbit is hunted by the fox.", "fox", "rabbit", "hunted", "passive"),
    ("The deer is attacked by the lion.", "lion", "deer", "attacked", "passive"),
    ("The meal is cooked by the chef.", "chef", "meal", "cooked", "passive"),
    
    # 不及物句
    ("The cat sleeps.", "cat", None, "sleeps", "intransitive"),
    ("The dog runs.", "dog", None, "runs", "intransitive"),
    ("The bird flies.", "bird", None, "flies", "intransitive"),
    ("The fish swims.", "fish", None, "swims", "intransitive"),
    ("The man walks.", "man", None, "walks", "intransitive"),
    ("The woman sings.", "woman", None, "sings", "intransitive"),
    ("The boy jumps.", "boy", None, "jumps", "intransitive"),
    ("The girl laughs.", "girl", None, "laughs", "intransitive"),
    ("The baby cries.", "baby", None, "cries", "intransitive"),
    ("The horse gallops.", "horse", None, "gallops", "intransitive"),
]

# 句法关系对: (sentence_1, sentence_2, relation_type)
SYNTAX_PAIRS = [
    # Active/Passive (same meaning, different structure)
    ("The cat chases the mouse.", "The mouse is chased by the cat.", "active_passive"),
    ("The dog bites the bone.", "The bone is bitten by the dog.", "active_passive"),
    ("The bird catches the fish.", "The fish is caught by the bird.", "active_passive"),
    ("The man reads the book.", "The book is read by the man.", "active_passive"),
    ("The boy throws the ball.", "The ball is thrown by the boy.", "active_passive"),
    
    # Tense change (same structure, different morphology)
    ("The cat chases the mouse.", "The cat chased the mouse.", "tense"),
    ("The dog bites the bone.", "The dog bit the bone.", "tense"),
    ("The bird catches the fish.", "The bird caught the fish.", "tense"),
    ("The man reads the book.", "The man read the book.", "tense"),
    ("The boy throws the ball.", "The boy threw the ball.", "tense"),
    
    # Plural change (same structure, different morphology)
    ("The cat chases the mouse.", "The cats chase the mice.", "plural"),
    ("The dog bites the bone.", "The dogs bite the bones.", "plural"),
    ("The bird catches the fish.", "The birds catch the fishes.", "plural"),
    ("The man reads the book.", "The men read the books.", "plural"),
    ("The boy throws the ball.", "The boys throw the balls.", "plural"),
]


# ============================================================
# 50A: Token级角色探测
# ============================================================
def exp_50a_token_level_role_probing(model, tokenizer, info, model_name):
    """
    核心问题: Phase 49B的agent探测=0.0是因为看了错误的token位置吗?
    
    方法: 从agent/patient/verb的token位置分别探测:
    1. 从agent位置能读出"我是agent"吗?
    2. 从patient位置能读出"我是patient"吗?
    3. 从verb位置能读出"我是动词"吗?
    """
    print("\n" + "="*70)
    print("50A: Token-Level Role Probing — 从token位置探测角色")
    print("="*70)
    
    n_layers = info.n_layers
    d_model = info.d_model
    device = next(model.parameters()).device
    
    sample_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    
    # 收集所有带标注句子的token级hidden states
    print("\n--- Collecting token-level hidden states ---")
    all_data = []  # [(sentence, agent_word, patient_word, verb_word, struct_type, 
                    #   {layer: h[seq_len, d_model]}, token_ids)]
    
    for sent, agent, patient, verb, stype in ANNOTATED_SENTENCES:
        h_dict, token_ids = get_all_token_hidden_states(model, tokenizer, sent, n_layers, device)
        all_data.append((sent, agent, patient, verb, stype, h_dict, token_ids))
        gc.collect()
    
    # --- Analysis 1: Agent token role probing ---
    print("\n--- Analysis 1: Agent Token Role Probing ---")
    print("  Can we linearly read 'I am the agent' from the agent token position?")
    
    # 构建数据集: agent位置 h vs non-agent位置 h
    # 标签: 1 = agent token, 0 = non-agent token
    for li in sample_layers:
        X_agent = []
        X_other = []
        
        for sent, agent, patient, verb, stype, h_dict, token_ids in all_data:
            if li not in h_dict:
                continue
            h = h_dict[li]  # [seq_len, d_model]
            agent_pos = find_token_position(token_ids, tokenizer, agent)
            
            if not agent_pos:
                continue
            
            # Agent token(s)
            for p in agent_pos:
                if p < h.shape[0]:
                    X_agent.append(h[p])
            
            # Non-agent tokens (exclude special tokens at pos 0)
            for p in range(1, h.shape[0]):
                if p not in agent_pos:
                    X_other.append(h[p])
        
        if len(X_agent) < 5 or len(X_other) < 5:
            print(f"  Layer {li}: Insufficient data (agent={len(X_agent)}, other={len(X_other)})")
            continue
        
        X_agent = np.array(X_agent)
        X_other = np.array(X_other)
        
        # 限制other数量，保持平衡
        if len(X_other) > len(X_agent) * 3:
            idx = np.random.choice(len(X_other), len(X_agent) * 3, replace=False)
            X_other = X_other[idx]
        
        X = np.vstack([X_agent, X_other])
        y = np.array([1]*len(X_agent) + [0]*len(X_other))
        
        # 归一化
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        X_norm = X / norms
        
        # 线性探测
        try:
            lr = LogisticRegression(max_iter=1000, C=1.0)
            scores_raw = cross_val_score(lr, X, y, cv=min(5, len(X)//10+2), scoring='accuracy')
            scores_norm = cross_val_score(lr, X_norm, y, cv=min(5, len(X)//10+2), scoring='accuracy')
            print(f"  Layer {li:3d}: Raw acc={scores_raw.mean():.3f}±{scores_raw.std():.3f}, "
                  f"Norm acc={scores_norm.mean():.3f}±{scores_norm.std():.3f} "
                  f"(N_agent={len(X_agent)}, N_other={len(X_other)})")
        except Exception as e:
            print(f"  Layer {li}: Error: {e}")
    
    # --- Analysis 2: Patient token role probing ---
    print("\n--- Analysis 2: Patient Token Role Probing ---")
    print("  Can we linearly read 'I am the patient' from the patient token position?")
    
    for li in sample_layers:
        X_patient = []
        X_other = []
        
        for sent, agent, patient, verb, stype, h_dict, token_ids in all_data:
            if patient is None:
                continue  # Skip intransitive
            if li not in h_dict:
                continue
            h = h_dict[li]
            patient_pos = find_token_position(token_ids, tokenizer, patient)
            
            if not patient_pos:
                continue
            
            for p in patient_pos:
                if p < h.shape[0]:
                    X_patient.append(h[p])
            
            agent_pos = find_token_position(token_ids, tokenizer, agent)
            for p in range(1, h.shape[0]):
                if p not in patient_pos and (not agent_pos or p not in agent_pos):
                    X_other.append(h[p])
        
        if len(X_patient) < 5 or len(X_other) < 5:
            continue
        
        X_patient = np.array(X_patient)
        X_other = np.array(X_other)
        
        if len(X_other) > len(X_patient) * 3:
            idx = np.random.choice(len(X_other), len(X_patient) * 3, replace=False)
            X_other = X_other[idx]
        
        X = np.vstack([X_patient, X_other])
        y = np.array([1]*len(X_patient) + [0]*len(X_other))
        
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        X_norm = X / norms
        
        try:
            lr = LogisticRegression(max_iter=1000, C=1.0)
            scores_raw = cross_val_score(lr, X, y, cv=min(5, len(X)//10+2), scoring='accuracy')
            scores_norm = cross_val_score(lr, X_norm, y, cv=min(5, len(X)//10+2), scoring='accuracy')
            print(f"  Layer {li:3d}: Raw acc={scores_raw.mean():.3f}±{scores_raw.std():.3f}, "
                  f"Norm acc={scores_norm.mean():.3f}±{scores_norm.std():.3f} "
                  f"(N_patient={len(X_patient)}, N_other={len(X_other)})")
        except Exception as e:
            print(f"  Layer {li}: Error: {e}")
    
    # --- Analysis 3: Three-way role classification ---
    print("\n--- Analysis 3: Three-way Role Classification ---")
    print("  Classify token as agent / patient / verb / other")
    
    for li in sample_layers:
        X_data = []
        y_data = []
        
        for sent, agent, patient, verb, stype, h_dict, token_ids in all_data:
            if li not in h_dict:
                continue
            h = h_dict[li]
            
            agent_pos = find_token_position(token_ids, tokenizer, agent)
            patient_pos = find_token_position(token_ids, tokenizer, patient) if patient else []
            verb_pos = find_token_position(token_ids, tokenizer, verb)
            
            for p in range(1, h.shape[0]):  # Skip BOS
                if p in agent_pos:
                    X_data.append(h[p])
                    y_data.append(0)  # agent
                elif p in patient_pos:
                    X_data.append(h[p])
                    y_data.append(1)  # patient
                elif p in verb_pos:
                    X_data.append(h[p])
                    y_data.append(2)  # verb
                else:
                    X_data.append(h[p])
                    y_data.append(3)  # other
        
        X = np.array(X_data)
        y = np.array(y_data)
        
        if len(np.unique(y)) < 4 or min(np.bincount(y)) < 5:
            print(f"  Layer {li}: Insufficient class diversity")
            continue
        
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        X_norm = X / norms
        
        try:
            lr = LogisticRegression(max_iter=1000, C=1.0)
            scores_raw = cross_val_score(lr, X, y, cv=min(5, len(X)//20+2), scoring='accuracy')
            scores_norm = cross_val_score(lr, X_norm, y, cv=min(5, len(X)//20+2), scoring='accuracy')
            
            # Per-class accuracy
            lr_raw = LogisticRegression(max_iter=1000, C=1.0).fit(X, y)
            lr_norm = LogisticRegression(max_iter=1000, C=1.0).fit(X_norm, y)
            
            print(f"  Layer {li:3d}: Raw acc={scores_raw.mean():.3f}, Norm acc={scores_norm.mean():.3f}")
        except Exception as e:
            print(f"  Layer {li}: Error: {e}")
    
    # --- Analysis 4: Agent token in active vs passive ---
    print("\n--- Analysis 4: Agent Token Consistency across Active/Passive ---")
    print("  Is the agent token representation similar in active vs passive?")
    
    # 找到匹配的active/passive对
    active_data = {}
    passive_data = {}
    
    for sent, agent, patient, verb, stype, h_dict, token_ids in all_data:
        if stype == "transitive":
            active_data[agent] = (h_dict, token_ids, sent)
        elif stype == "passive":
            passive_data[agent] = (h_dict, token_ids, sent)
    
    for li in sample_layers:
        cos_sims = []
        for agent in active_data:
            if agent not in passive_data:
                continue
            ah, atoks, asent = active_data[agent]
            ph, ptoks, psent = passive_data[agent]
            
            if li not in ah or li not in ph:
                continue
            
            a_pos = find_token_position(atoks, tokenizer, agent)
            p_pos = find_token_position(ptoks, tokenizer, agent)
            
            if not a_pos or not p_pos:
                continue
            
            a_h = ah[li][a_pos[0]]
            p_h = ph[li][p_pos[0]]
            
            cos_sim = np.dot(a_h, p_h) / (np.linalg.norm(a_h) * np.linalg.norm(p_h) + 1e-8)
            cos_sims.append(cos_sim)
        
        if cos_sims:
            print(f"  Layer {li:3d}: Agent token cos(active,passive) = {np.mean(cos_sims):.3f} ± {np.std(cos_sims):.3f} (N={len(cos_sims)})")
    
    print("\n" + "="*70)
    print("50A SUMMARY: Token-Level Role Probing")
    print("="*70)


# ============================================================
# 50B: 跨Token关系提取
# ============================================================
def exp_50b_cross_token_relations(model, tokenizer, info, model_name):
    """
    核心问题: h_i - h_j 是否编码语法关系(agent-verb, patient-verb)?
    
    方法:
    1. 对每个句子，计算 agent-verb, patient-verb, agent-patient 的差向量
    2. 测试这些差向量能否区分语法关系类型
    3. 测试差向量是否在语法变换下保持不变
    """
    print("\n" + "="*70)
    print("50B: Cross-Token Relation Extraction — 跨Token关系提取")
    print("="*70)
    
    n_layers = info.n_layers
    d_model = info.d_model
    device = next(model.parameters()).device
    
    sample_layers = [n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    
    # 收集数据
    print("\n--- Collecting token-level data for relation analysis ---")
    all_data = []
    for item in ANNOTATED_SENTENCES[:30]:
        sent, agent, patient, verb, stype = item
        h_dict, token_ids = get_all_token_hidden_states(model, tokenizer, sent, n_layers, device)
        all_data.append((sent, agent, patient, verb, stype, h_dict, token_ids))
        gc.collect()
    
    # --- Analysis 1: Relation vector PCA ---
    print("\n--- Analysis 1: Relation Vector PCA ---")
    print("  What is the dimensionality of agent-verb relation vectors?")
    
    for li in sample_layers:
        av_vecs = []  # agent-verb
        pv_vecs = []  # patient-verb
        ap_vecs = []  # agent-patient
        
        for sent, agent, patient, verb, stype, h_dict, token_ids in all_data:
            if patient is None:
                continue
            if li not in h_dict:
                continue
            h = h_dict[li]
            
            a_pos = find_token_position(token_ids, tokenizer, agent)
            p_pos = find_token_position(token_ids, tokenizer, patient)
            v_pos = find_token_position(token_ids, tokenizer, verb)
            
            if not a_pos or not p_pos or not v_pos:
                continue
            
            a_h = h[a_pos[0]]
            p_h = h[p_pos[0]]
            v_h = h[v_pos[0]]
            
            av_vecs.append(a_h - v_h)
            pv_vecs.append(p_h - v_h)
            ap_vecs.append(a_h - p_h)
        
        if len(av_vecs) < 5:
            print(f"  Layer {li}: Insufficient data")
            continue
        
        av_vecs = np.array(av_vecs)
        pv_vecs = np.array(pv_vecs)
        ap_vecs = np.array(ap_vecs)
        
        # PCA on relation vectors
        pca_av = PCA()
        pca_av.fit(av_vecs)
        pca_pv = PCA()
        pca_pv.fit(pv_vecs)
        pca_ap = PCA()
        pca_ap.fit(ap_vecs)
        
        # Cumulative variance
        cum_av = np.cumsum(pca_av.explained_variance_ratio_)
        cum_pv = np.cumsum(pca_pv.explained_variance_ratio_)
        cum_ap = np.cumsum(pca_ap.explained_variance_ratio_)
        
        dim90_av = np.searchsorted(cum_av, 0.9) + 1
        dim90_pv = np.searchsorted(cum_pv, 0.9) + 1
        dim90_ap = np.searchsorted(cum_ap, 0.9) + 1
        
        print(f"  Layer {li}:")
        print(f"    Agent-Verb: dim90={dim90_av}, top1={pca_av.explained_variance_ratio_[0]:.3f}, top5={cum_av[min(4,len(cum_av)-1)]:.3f}")
        print(f"    Patient-Verb: dim90={dim90_pv}, top1={pca_pv.explained_variance_ratio_[0]:.3f}, top5={cum_pv[min(4,len(cum_pv)-1)]:.3f}")
        print(f"    Agent-Patient: dim90={dim90_ap}, top1={pca_ap.explained_variance_ratio_[0]:.3f}, top5={cum_ap[min(4,len(cum_ap)-1)]:.3f}")
    
    # --- Analysis 2: Relation type discrimination ---
    print("\n--- Analysis 2: Can relation vectors discriminate syntax? ---")
    print("  Can we tell agent-verb from patient-verb relation vectors?")
    
    for li in sample_layers:
        av_vecs = []
        pv_vecs = []
        
        for sent, agent, patient, verb, stype, h_dict, token_ids in all_data:
            if patient is None:
                continue
            if li not in h_dict:
                continue
            h = h_dict[li]
            
            a_pos = find_token_position(token_ids, tokenizer, agent)
            p_pos = find_token_position(token_ids, tokenizer, patient)
            v_pos = find_token_position(token_ids, tokenizer, verb)
            
            if not a_pos or not p_pos or not v_pos:
                continue
            
            av_vecs.append(h[a_pos[0]] - h[v_pos[0]])
            pv_vecs.append(h[p_pos[0]] - h[v_pos[0]])
        
        if len(av_vecs) < 5:
            continue
        
        X = np.vstack([av_vecs, pv_vecs])
        y = np.array([0]*len(av_vecs) + [1]*len(pv_vecs))
        
        # Raw and normalized
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        X_norm = X / norms
        
        try:
            lr = LogisticRegression(max_iter=1000, C=1.0)
            scores_raw = cross_val_score(lr, X, y, cv=min(5, len(X)//10+2), scoring='accuracy')
            scores_norm = cross_val_score(lr, X_norm, y, cv=min(5, len(X)//10+2), scoring='accuracy')
            
            # LDA
            lda = LinearDiscriminantAnalysis()
            lda.fit(X, y)
            lda_acc = lda.score(X, y)
            
            print(f"  Layer {li:3d}: Raw acc={scores_raw.mean():.3f}, Norm acc={scores_norm.mean():.3f}, LDA acc={lda_acc:.3f}")
        except Exception as e:
            print(f"  Layer {li}: Error: {e}")
    
    # --- Analysis 3: Relation invariance under syntax ---
    print("\n--- Analysis 3: Relation Invariance under Active/Passive ---")
    print("  Is agent-verb relation invariant to voice change?")
    
    for pair_a, pair_b, rtype in SYNTAX_PAIRS[:5]:  # Only active/passive
        if rtype != "active_passive":
            continue
        
        h_a, toks_a = get_all_token_hidden_states(model, tokenizer, pair_a, n_layers, device)
        h_b, toks_b = get_all_token_hidden_states(model, tokenizer, pair_b, n_layers, device)
        
        # Find agent and verb in both
        # Active: "The cat chases the mouse" → agent=cat, verb=chases
        # Passive: "The mouse is chased by the cat" → agent=cat, verb=chased
        
        # Use first noun as agent in active, find it in passive
        # This is approximate - we look for the main content words
        for li in sample_layers:
            if li not in h_a or li not in h_b:
                continue
            
            ha = h_a[li]  # [seq_len, d_model]
            hb = h_b[li]
            
            # Compute all pairwise cosine similarities between token pairs
            # Find the best matching token pairs
            n_a = ha.shape[0]
            n_b = hb.shape[0]
            
            # Cosine similarity matrix between tokens of sentence A and B
            ha_norm = ha / (np.linalg.norm(ha, axis=1, keepdims=True) + 1e-8)
            hb_norm = hb / (np.linalg.norm(hb, axis=1, keepdims=True) + 1e-8)
            cos_mat = ha_norm @ hb_norm.T  # [n_a, n_b]
            
            # Average cross-sentence similarity
            avg_cos = cos_mat[1:, 1:].mean()  # exclude BOS
        
        gc.collect()
    
    # More systematic: compute relation invariance for matched pairs
    print("\n  Systematic analysis:")
    for li in sample_layers:
        active_av = []  # agent-verb in active
        passive_av = []  # agent-verb in passive (agent is after "by")
        
        for sent, agent, patient, verb, stype, h_dict, token_ids in all_data:
            if li not in h_dict:
                continue
            h = h_dict[li]
            
            a_pos = find_token_position(token_ids, tokenizer, agent)
            v_pos = find_token_position(token_ids, tokenizer, verb)
            
            if not a_pos or not v_pos:
                continue
            
            rel_vec = h[a_pos[0]] - h[v_pos[0]]
            
            if stype == "transitive":
                active_av.append(rel_vec)
            elif stype == "passive":
                passive_av.append(rel_vec)
        
        if len(active_av) < 3 or len(passive_av) < 3:
            continue
        
        active_av = np.array(active_av)
        passive_av = np.array(passive_av)
        
        # Cosine similarity between active and passive agent-verb relations
        a_norm = active_av / (np.linalg.norm(active_av, axis=1, keepdims=True) + 1e-8)
        p_norm = passive_av / (np.linalg.norm(passive_av, axis=1, keepdims=True) + 1e-8)
        
        # Mean pairwise cosine between active and passive
        cross_cos = a_norm @ p_norm.T
        mean_cross = cross_cos.mean()
        
        # Within-group cosines
        if len(a_norm) > 1:
            within_a = (a_norm @ a_norm.T)
            np.fill_diagonal(within_a, 0)
            within_a_mean = within_a[within_a != 0].mean()
        else:
            within_a_mean = 0
        
        if len(p_norm) > 1:
            within_p = (p_norm @ p_norm.T)
            np.fill_diagonal(within_p, 0)
            within_p_mean = within_p[within_p != 0].mean()
        else:
            within_p_mean = 0
        
        print(f"  Layer {li:3d}: Cross voice cos={mean_cross:.3f}, "
              f"Within active={within_a_mean:.3f}, Within passive={within_p_mean:.3f}")
    
    # --- Analysis 4: Cross-token binding signature ---
    print("\n--- Analysis 4: Cross-Token Binding Signature ---")
    print("  Is there a consistent 'binding pattern' between agent and verb?")
    
    for li in sample_layers:
        # For each agent word, compute the change in verb representation
        # when paired with different agents
        # e.g., h(cat chases) - h(cat) vs h(dog chases) - h(dog)
        
        agent_h = defaultdict(list)  # agent_word -> list of h at agent position
        verb_h = defaultdict(list)   # verb_word -> list of h at verb position
        
        for sent, agent, patient, verb, stype, h_dict, token_ids in all_data:
            if li not in h_dict:
                continue
            h = h_dict[li]
            
            a_pos = find_token_position(token_ids, tokenizer, agent)
            v_pos = find_token_position(token_ids, tokenizer, verb)
            
            if a_pos and a_pos[0] < h.shape[0]:
                agent_h[agent].append(h[a_pos[0]])
            if v_pos and v_pos[0] < h.shape[0]:
                verb_h[verb].append(h[v_pos[0]])
        
        # Compute average agent representation
        agent_means = {}
        for agent, vecs in agent_h.items():
            if len(vecs) >= 1:
                agent_means[agent] = np.mean(vecs, axis=0)
        
        # Test: is the same agent similar across different sentences?
        if len(agent_means) >= 3:
            agents = list(agent_means.keys())
            agent_mat = np.array([agent_means[a] for a in agents])
            agent_mat_norm = agent_mat / (np.linalg.norm(agent_mat, axis=1, keepdims=True) + 1e-8)
            cos_mat = agent_mat_norm @ agent_mat_norm.T
            np.fill_diagonal(cos_mat, 0)
            mean_cos = cos_mat[cos_mat != 0].mean()
            print(f"  Layer {li:3d}: Agent consistency across sentences: mean_cos={mean_cos:.3f} (N_agents={len(agents)})")
    
    print("\n" + "="*70)
    print("50B SUMMARY: Cross-Token Relations")
    print("="*70)


# ============================================================
# 50C: 注意力模式分析
# ============================================================
def exp_50c_attention_patterns(model, tokenizer, info, model_name):
    """
    核心问题: 注意力矩阵是否编码语法关系(agent→verb, patient→verb)?
    
    方法:
    1. 提取各层注意力权重
    2. 分析agent/verb/patient位置间的注意力模式
    3. 测试注意力是否区分语法角色
    """
    print("\n" + "="*70)
    print("50C: Attention Pattern Analysis — 注意力模式分析")
    print("="*70)
    
    n_layers = info.n_layers
    d_model = info.d_model
    device = next(model.parameters()).device
    
    sample_layers = [n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    
    # --- Analysis 1: Token-level attention from agent ---
    print("\n--- Analysis 1: Where does the agent token attend? ---")
    
    test_sentences = ANNOTATED_SENTENCES[:20]  # Use transitive + passive only
    
    for li in sample_layers:
        agent_to_verb_attn = []
        agent_to_patient_attn = []
        agent_to_other_attn = []
        
        for sent, agent, patient, verb, stype in test_sentences:
            if patient is None:
                continue
            
            # Run model with attention outputs
            inputs = tokenizer(sent, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]
            
            with torch.no_grad():
                # Try to get attention weights using output_attentions
                embed = model.get_input_embeddings()(input_ids)
                position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
                
                try:
                    outputs = model(inputs_embeds=embed, position_ids=position_ids, 
                                   output_attentions=True)
                    attn = outputs.attentions  # tuple of [1, n_heads, seq, seq]
                    
                    if attn is None or len(attn) == 0:
                        continue
                    
                    attn_l = attn[li].detach().cpu().numpy()[0]  # [n_heads, seq, seq]
                    token_ids_np = input_ids[0].cpu().numpy()
                    
                    a_pos = find_token_position(token_ids_np, tokenizer, agent)
                    p_pos = find_token_position(token_ids_np, tokenizer, patient)
                    v_pos = find_token_position(token_ids_np, tokenizer, verb)
                    
                    if not a_pos or not p_pos or not v_pos:
                        continue
                    
                    # Average attention from agent to verb, patient, other
                    a_idx = a_pos[0]
                    
                    # Mean across heads
                    mean_attn = attn_l.mean(axis=0)  # [seq, seq]
                    
                    a2v = mean_attn[a_idx, v_pos[0]]
                    a2p = mean_attn[a_idx, p_pos[0]]
                    
                    # Other = all positions except agent, verb, patient
                    other_attn = []
                    for j in range(mean_attn.shape[1]):
                        if j not in a_pos and j not in v_pos and j not in p_pos:
                            other_attn.append(mean_attn[a_idx, j])
                    a2o = np.mean(other_attn) if other_attn else 0
                    
                    agent_to_verb_attn.append(a2v)
                    agent_to_patient_attn.append(a2p)
                    agent_to_other_attn.append(a2o)
                    
                except Exception as e:
                    continue
            
            del embed
            gc.collect()
        
        if agent_to_verb_attn:
            print(f"  Layer {li:3d}: Agent→Verb={np.mean(agent_to_verb_attn):.4f}, "
                  f"Agent→Patient={np.mean(agent_to_patient_attn):.4f}, "
                  f"Agent→Other={np.mean(agent_to_other_attn):.4f}")
    
    # --- Analysis 2: Attention-based role classification ---
    print("\n--- Analysis 2: Can attention patterns predict syntax? ---")
    
    # For each sentence, extract a summary of its attention pattern
    # Then classify as transitive/intransitive/passive
    
    for li in sample_layers:
        features = []
        labels = []
        
        for sent, agent, patient, verb, stype in test_sentences:
            inputs = tokenizer(sent, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]
            
            with torch.no_grad():
                embed = model.get_input_embeddings()(input_ids)
                position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
                
                try:
                    outputs = model(inputs_embeds=embed, position_ids=position_ids, 
                                   output_attentions=True)
                    attn = outputs.attentions
                    
                    if attn is None:
                        continue
                    
                    attn_l = attn[li].detach().cpu().numpy()[0]  # [n_heads, seq, seq]
                    mean_attn = attn_l.mean(axis=0)  # [seq, seq]
                    
                    # Feature: attention entropy, diagonal dominance, etc.
                    # Entropy of attention distribution
                    n = mean_attn.shape[0]
                    entropy = -np.sum(mean_attn * np.log(mean_attn + 1e-10)) / n
                    
                    # Diagonal attention (self-attention)
                    diag_attn = np.mean(np.diag(mean_attn))
                    
                    # Off-diagonal attention
                    off_diag = (mean_attn.sum() - np.diag(mean_attn).sum()) / (n*n - n + 1e-8)
                    
                    # Attention concentration (max attention per row)
                    max_attn = np.mean(np.max(mean_attn, axis=1))
                    
                    features.append([entropy, diag_attn, off_diag, max_attn])
                    labels.append(stype)
                    
                except Exception as e:
                    continue
            
            del embed
            gc.collect()
        
        if len(features) < 10:
            continue
        
        X = np.array(features)
        y = np.array(labels)
        
        try:
            lr = LogisticRegression(max_iter=1000, C=1.0, multi_class='ovr')
            scores = cross_val_score(lr, X, y, cv=min(5, len(X)//5+2), scoring='accuracy')
            print(f"  Layer {li:3d}: Syntax type from attention features: acc={scores.mean():.3f}±{scores.std():.3f}")
        except Exception as e:
            print(f"  Layer {li}: Error: {e}")
    
    print("\n" + "="*70)
    print("50C SUMMARY: Attention Patterns")
    print("="*70)


# ============================================================
# 50D: 关系图恢复
# ============================================================
def exp_50d_relation_graph_recovery(model, tokenizer, info, model_name):
    """
    核心问题: 从token间相似度矩阵能恢复语法树吗?
    
    方法:
    1. 计算token间的cosine相似度矩阵
    2. 构建最小生成树(MST)
    3. 比较MST与语法依赖树
    4. 测试归一化后的表示是否给出更好的树
    """
    print("\n" + "="*70)
    print("50D: Relation Graph Recovery — 关系图恢复")
    print("="*70)
    
    n_layers = info.n_layers
    d_model = info.d_model
    device = next(model.parameters()).device
    
    sample_layers = [n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    
    # 选取几个典型句子进行详细分析
    test_cases = [
        ("The cat chases the mouse.", {"The": "det", "cat": "agent", "chases": "verb", "mouse": "patient"}),
        ("The mouse is chased by the cat.", {"The": "det", "mouse": "patient", "is": "aux", "chased": "verb", "cat": "agent", "by": "prep"}),
        ("The man reads the book.", {"The": "det", "man": "agent", "reads": "verb", "book": "patient"}),
        ("The cat sleeps.", {"The": "det", "cat": "agent", "sleeps": "verb"}),
    ]
    
    for li in sample_layers:
        print(f"\n  --- Layer {li} ---")
        
        for sent, roles in test_cases:
            h_dict, token_ids = get_all_token_hidden_states(model, tokenizer, sent, n_layers, device)
            
            if li not in h_dict:
                continue
            
            h = h_dict[li]  # [seq_len, d_model]
            seq_len = h.shape[0]
            
            # Decode tokens
            tokens = [safe_decode(tokenizer, tid) for tid in token_ids]
            
            # Compute cosine similarity matrix
            h_norm = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-8)
            cos_mat = h_norm @ h_norm.T  # [seq_len, seq_len]
            
            # Build distance matrix for MST
            dist_mat = 1 - cos_mat
            np.fill_diagonal(dist_mat, 0)
            
            # Simple MST using greedy approach
            # (We don't import scipy for this, use a simple approach)
            # Instead, for each token, find its most similar other token
            print(f"\n    Sentence: {sent}")
            print(f"    Tokens: {tokens}")
            print(f"    Most-similar pairs (cosine):")
            
            # Find top-3 pairs (excluding diagonal and duplicates)
            pairs = []
            for i in range(1, seq_len):  # skip BOS
                for j in range(i+1, seq_len):
                    pairs.append((i, j, cos_mat[i, j]))
            pairs.sort(key=lambda x: -x[2])
            
            for i, j, c in pairs[:5]:
                print(f"      {tokens[i]:>12} ↔ {tokens[j]:>12}: cos={c:.3f}")
            
            # Check: do content words cluster together?
            content_pos = [i for i in range(1, seq_len) 
                          if any(w.lower() in tokens[i].lower() for w in roles.keys() 
                                if roles[w] in ['agent', 'patient', 'verb'])]
            
            if len(content_pos) >= 2:
                content_cos = []
                for i in range(len(content_pos)):
                    for j in range(i+1, len(content_pos)):
                        content_cos.append(cos_mat[content_pos[i], content_pos[j]])
                func_cos = []
                for i in range(1, seq_len):
                    if i not in content_pos:
                        for j in range(i+1, seq_len):
                            if j not in content_pos:
                                func_cos.append(cos_mat[i, j])
                
                if content_cos and func_cos:
                    print(f"    Content-word avg cos: {np.mean(content_cos):.3f}")
                    print(f"    Function-word avg cos: {np.mean(func_cos):.3f}")
            
            gc.collect()
    
    # --- Analysis 2: Can we recover agent-verb edges? ---
    print("\n--- Analysis 2: Agent-Verb Edge Recovery ---")
    print("  For each transitive sentence, is the agent-verb pair the most similar?")
    
    correct_count = 0
    total_count = 0
    
    for li in [n_layers//2, n_layers-1]:
        for sent, agent, patient, verb, stype in ANNOTATED_SENTENCES[:20]:
            if patient is None or stype != "transitive":
                continue
            
            h_dict, token_ids = get_all_token_hidden_states(model, tokenizer, sent, n_layers, device)
            
            if li not in h_dict:
                continue
            
            h = h_dict[li]
            seq_len = h.shape[0]
            tokens = [safe_decode(tokenizer, tid) for tid in token_ids]
            
            h_norm = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-8)
            cos_mat = h_norm @ h_norm.T
            
            a_pos = find_token_position(token_ids, tokenizer, agent)
            v_pos = find_token_position(token_ids, tokenizer, verb)
            
            if not a_pos or not v_pos:
                continue
            
            a_idx = a_pos[0]
            v_idx = v_pos[0]
            
            # What does the agent attend most to (by cosine similarity)?
            agent_sims = cos_mat[a_idx, 1:]  # exclude BOS
            best_match = np.argmax(agent_sims) + 1  # +1 because we excluded index 0
            
            is_verb = (best_match == v_idx)
            correct_count += is_verb
            total_count += 1
            
            gc.collect()
        
        if total_count > 0:
            print(f"  Layer {li:3d}: Agent's most similar token is verb: {correct_count}/{total_count} = {correct_count/total_count:.1%}")
    
    print("\n" + "="*70)
    print("50D SUMMARY: Relation Graph Recovery")
    print("="*70)


# ============================================================
# 50E: 全局上下文操作
# ============================================================
def exp_50e_context_dependent_operations(model, tokenizer, info, model_name):
    """
    核心问题: Δz = f(z, context)? 操作是否依赖全局上下文?
    
    方法:
    1. 同一token在不同上下文中的Δz是否不同?
    2. 引入context特征(其他token的h)后，Δz预测是否改善?
    3. 上下文依赖的强度随层变化?
    """
    print("\n" + "="*70)
    print("50E: Context-Dependent Operations — 全局上下文操作")
    print("="*70)
    
    n_layers = info.n_layers
    d_model = info.d_model
    device = next(model.parameters()).device
    
    sample_layers = [n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    
    # --- Analysis 1: Same word, different context ---
    print("\n--- Analysis 1: Same Word, Different Context ---")
    print("  How much does the representation of 'cat' change across contexts?")
    
    # Context pairs: same word in different sentences
    context_pairs = [
        ("The cat sits on the mat.", "cat", "resting"),
        ("The cat chases the mouse.", "cat", "agent"),
        ("The cat is chased by the dog.", "cat", "patient"),
        ("The big cat runs fast.", "cat", "modified"),
        ("The cat and the dog play.", "cat", "conjunction"),
        ("The dog bites the bone.", "dog", "agent"),
        ("The dog is bitten by the cat.", "dog", "patient"),
        ("The old dog walks slowly.", "dog", "modified"),
        ("The bird catches the fish.", "bird", "agent"),
        ("The fish is caught by the bird.", "fish", "patient"),
    ]
    
    # Collect h for target word in each context
    for li in sample_layers:
        word_vecs = defaultdict(list)
        word_contexts = defaultdict(list)
        
        for sent, word, context in context_pairs:
            h_dict, token_ids = get_all_token_hidden_states(model, tokenizer, sent, n_layers, device)
            
            if li not in h_dict:
                continue
            h = h_dict[li]
            
            w_pos = find_token_position(token_ids, tokenizer, word)
            if w_pos and w_pos[0] < h.shape[0]:
                word_vecs[word].append(h[w_pos[0]])
                word_contexts[word].append(context)
            
            gc.collect()
        
        # For each word with multiple contexts, compute inter-context similarity
        print(f"\n  Layer {li}:")
        for word, vecs in word_vecs.items():
            if len(vecs) < 2:
                continue
            vecs = np.array(vecs)
            vecs_norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
            cos_mat = vecs_norm @ vecs_norm.T
            
            # Mean pairwise cosine
            n = len(vecs)
            total_cos = 0
            count = 0
            for i in range(n):
                for j in range(i+1, n):
                    total_cos += cos_mat[i, j]
                    count += 1
            
            if count > 0:
                contexts = word_contexts[word]
                print(f"    '{word}' across {n} contexts: mean_cos={total_cos/count:.3f}")
                for i in range(n):
                    print(f"      Context: {contexts[i]}")
    
    # --- Analysis 2: Context-augmented Δz prediction ---
    print("\n--- Analysis 2: Context-Augmented Δz Prediction ---")
    print("  Does adding context (mean of other tokens) improve Δz prediction?")
    
    # Use last-token Δz as before, but now include context features
    for li in sample_layers:
        # Collect data
        z_prev_list = []
        dz_list = []
        z_context_list = []
        
        for sent, agent, patient, verb, stype in ANNOTATED_SENTENCES[:30]:
            h_dict, token_ids = get_all_token_hidden_states(model, tokenizer, sent, n_layers, device)
            
            if li not in h_dict or (li-1) not in h_dict:
                continue
            
            h_curr = h_dict[li][-1]  # last token at layer li
            h_prev = h_dict[li-1][-1]  # last token at layer li-1
            
            # Context = mean of all other tokens
            context = h_dict[li-1][1:-1].mean(axis=0) if h_dict[li-1].shape[0] > 2 else h_dict[li-1][-1]
            
            z_prev_list.append(h_prev)
            dz_list.append(h_curr - h_prev)
            z_context_list.append(context)
            
            gc.collect()
        
        if len(z_prev_list) < 10:
            continue
        
        z_prev = np.array(z_prev_list)
        dz = np.array(dz_list)
        z_context = np.array(z_context_list)
        
        # Normalize
        z_prev_norm = z_prev / (np.linalg.norm(z_prev, axis=1, keepdims=True) + 1e-8)
        dz_norm = dz / (np.linalg.norm(dz, axis=1, keepdims=True) + 1e-8)
        z_context_norm = z_context / (np.linalg.norm(z_context, axis=1, keepdims=True) + 1e-8)
        
        # Baseline: Δz ~ z (no context)
        ridge_base = Ridge(alpha=1.0)
        ridge_base.fit(z_prev_norm, dz_norm)
        r2_base = ridge_base.score(z_prev_norm, dz_norm)
        
        # With context: Δz ~ [z, context]
        X_aug = np.hstack([z_prev_norm, z_context_norm])
        ridge_aug = Ridge(alpha=1.0)
        ridge_aug.fit(X_aug, dz_norm)
        r2_aug = ridge_aug.score(X_aug, dz_norm)
        
        print(f"  Layer {li:3d}: R²(z→Δz)={r2_base:.3f}, R²([z,ctx]→Δz)={r2_aug:.3f}, Δ={r2_aug-r2_base:+.3f}")
    
    # --- Analysis 3: Token-specific Δz patterns ---
    print("\n--- Analysis 3: Token-Specific Δz Patterns ---")
    print("  Is Δz for the same token type consistent across sentences?")
    
    # Compare Δz for verb tokens across different sentences
    for li in sample_layers:
        verb_dz = defaultdict(list)
        
        for sent, agent, patient, verb, stype in ANNOTATED_SENTENCES[:20]:
            h_dict, token_ids = get_all_token_hidden_states(model, tokenizer, sent, n_layers, device)
            
            if li not in h_dict or (li-1) not in h_dict:
                continue
            
            h_curr = h_dict[li]
            h_prev = h_dict[li-1]
            
            v_pos = find_token_position(token_ids, tokenizer, verb)
            if v_pos and v_pos[0] < h_curr.shape[0]:
                dz_verb = h_curr[v_pos[0]] - h_prev[v_pos[0]]
                verb_dz[verb].append(dz_verb)
            
            gc.collect()
        
        # For verbs appearing multiple times
        for verb, dz_list in verb_dz.items():
            if len(dz_list) < 2:
                continue
            dz_arr = np.array(dz_list)
            dz_norm = dz_arr / (np.linalg.norm(dz_arr, axis=1, keepdims=True) + 1e-8)
            
            cos_mat = dz_norm @ dz_norm.T
            n = len(dz_list)
            total_cos = 0
            count = 0
            for i in range(n):
                for j in range(i+1, n):
                    total_cos += cos_mat[i, j]
                    count += 1
            
            if count > 0 and li == sample_layers[0]:  # Only print for first layer to avoid spam
                print(f"  Layer {li}, verb '{verb}': Δz consistency across {n} contexts: cos={total_cos/count:.3f}")
    
    print("\n" + "="*70)
    print("50E SUMMARY: Context-Dependent Operations")
    print("="*70)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Phase 50: Relational Structure Recovery")
    parser.add_argument("--model", type=str, default="deepseek7b",
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, default=0,
                       help="Experiment: 0=all, 1=50A, 2=50B, 3=50C, 4=50D, 5=50E")
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"Phase 50: Relational Structure Recovery — 关系结构恢复")
    print(f"Model: {args.model}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, args.model)
    
    print(f"  Model: {info.name}, Layers: {info.n_layers}, d_model: {info.d_model}")
    
    results = {}
    
    try:
        if args.exp in [0, 1]:
            results["50A"] = exp_50a_token_level_role_probing(model, tokenizer, info, args.model)
        
        if args.exp in [0, 2]:
            results["50B"] = exp_50b_cross_token_relations(model, tokenizer, info, args.model)
        
        if args.exp in [0, 3]:
            results["50C"] = exp_50c_attention_patterns(model, tokenizer, info, args.model)
        
        if args.exp in [0, 4]:
            results["50D"] = exp_50d_relation_graph_recovery(model, tokenizer, info, args.model)
        
        if args.exp in [0, 5]:
            results["50E"] = exp_50e_context_dependent_operations(model, tokenizer, info, args.model)
    
    finally:
        release_model(model)
    
    print(f"\n{'='*70}")
    print("Phase 50 COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
