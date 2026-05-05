"""
Phase 51: Semantic-Role Separation — 语义-角色分离
======================================================

SLT v2.0的核心错误: "角色信息不存在于单token向量中"
Phase 50纠正: 角色信息在TOKEN位置上线性可读(90%+)

但Phase 50有硬伤: agent探测90%+可能是词类效应而非角色效应
  - agent词几乎都是名词 → 检测的可能是"这是名词"不是"这是施事"
  - 需要控制词类变量

Phase 51核心任务:
  51A: 词类控制的角色探测 — 只在名词中区分agent vs patient
  51B: 角色向量构造 — h(agent位置) - h(同词patient位置) → 纯角色编码
  51C: 角色向量代数 — agent_vec + patient_vec = ? 角色向量可加吗?
  51D: 角色向量变换 — 主动→被动时角色向量如何变化?
  51E: 语义-角色正交性 — 语义方向和角色方向是否正交?

理论预测:
  如果SLT v2.0正确(角色=关系属性): 词类控制后角色探测→0
  如果修正理论正确(角色=局部属性): 词类控制后角色探测仍然>随机
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
from scipy.spatial.distance import cosine as scipy_cosine
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from model_utils import (load_model, get_layers, get_model_info, release_model,
                          safe_decode, collect_layer_outputs, get_W_U)


# ============================================================
# 辅助函数
# ============================================================
def get_all_token_hidden_states(model, tokenizer, sentence, n_layers, device):
    """收集句子所有token在所有层的hidden states"""
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
            result[li] = outputs[key][0].numpy()
    
    del outputs, embed
    return result, input_ids[0].cpu().numpy()


def find_token_position(token_ids, tokenizer, target_word):
    """找到目标词在token序列中的位置"""
    decoded = [safe_decode(tokenizer, tid) for tid in token_ids]
    positions = []
    for i, d in enumerate(decoded):
        if target_word.lower() in d.lower().strip():
            positions.append(i)
    if positions:
        return positions
    for i, d in enumerate(decoded):
        if target_word.lower() in d.lower():
            positions.append(i)
    return positions


def find_all_content_positions(token_ids, tokenizer):
    """找到所有内容词位置（名词、动词、形容词），排除功能词"""
    decoded = [safe_decode(tokenizer, tid) for tid in token_ids]
    function_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'by', 'in', 'on', 'at',
                      'to', 'for', 'with', 'and', 'or', 'but', 'of', '.', ',', '!', '?', 
                      '<|begin_of_sentence|>', '<｜begin▁of▁sentence｜>'}
    content_pos = []
    for i, d in enumerate(decoded):
        if d.strip().lower() not in function_words and i > 0:
            content_pos.append(i)
    return content_pos


# ============================================================
# 关键句子集：同一名词既作agent又作patient
# ============================================================
# ★ 核心设计: 同一个词在agent位置和patient位置
# 这才能控制词类变量，真正测试"角色"vs"词类"
ROLE_SWAP_PAIRS = [
    # 同一名词既作agent又作patient
    # (agent_sentence, patient_sentence, word, verb_in_agent, verb_in_patient)
    ("The cat chases the dog.", "The dog is chased by the cat.", "cat", "dog"),
    ("The cat watches the bird.", "The bird is watched by the cat.", "cat", "bird"),
    ("The dog follows the cat.", "The cat is followed by the dog.", "dog", "cat"),
    ("The man reads the book.", "The book is read by the man.", "man", "book"),
    ("The woman opens the door.", "The door is opened by the woman.", "woman", "door"),
    ("The boy throws the ball.", "The ball is thrown by the boy.", "boy", "ball"),
    ("The teacher writes the letter.", "The letter is written by the teacher.", "teacher", "letter"),
    ("The cat bites the dog.", "The dog is bitten by the cat.", "cat", "dog"),
    ("The lion attacks the deer.", "The deer is attacked by the lion.", "lion", "deer"),
    ("The fox hunts the rabbit.", "The rabbit is hunted by the fox.", "fox", "rabbit"),
    ("The bear eats the fish.", "The fish is eaten by the bear.", "bear", "fish"),
    ("The wolf chases the sheep.", "The sheep is chased by the wolf.", "wolf", "sheep"),
]

# 更丰富的"同词不同角色"集
SAME_WORD_DIFFERENT_ROLE = [
    # (sentence, target_word, role, sentence_type)
    ("The cat chases the mouse.", "cat", "agent", "active"),
    ("The mouse is chased by the cat.", "mouse", "agent_by", "passive"),  # cat是语义agent但语法上在by后
    ("The dog bites the bone.", "dog", "agent", "active"),
    ("The bone is bitten by the dog.", "bone", "agent_by", "passive"),
    ("The bird catches the fish.", "bird", "agent", "active"),
    ("The fish is caught by the bird.", "fish", "agent_by", "passive"),
    # 关键: 同一个名词作为agent和作为patient
    ("The cat chases the dog.", "cat", "agent", "active"),
    ("The dog chases the cat.", "cat", "patient", "active"),  # ★ 同一个cat在patient位置
    ("The dog bites the cat.", "cat", "patient", "active"),
    ("The man reads the book.", "man", "agent", "active"),
    ("The man is helped by the doctor.", "man", "patient", "passive"),  # man作为patient
    ("The boy throws the ball.", "boy", "agent", "active"),
    ("The girl hits the boy.", "boy", "patient", "active"),  # boy作为patient
    ("The teacher teaches the student.", "teacher", "agent", "active"),
    ("The principal praises the teacher.", "teacher", "patient", "active"),  # teacher作为patient
    ("The doctor helps the patient.", "doctor", "agent", "active"),
    ("The nurse calls the doctor.", "doctor", "patient", "active"),  # doctor作为patient
    ("The writer writes the book.", "writer", "agent", "active"),
    ("The editor critiques the writer.", "writer", "patient", "active"),  # writer作为patient
    ("The chef cooks the meal.", "chef", "agent", "active"),
    ("The critic reviews the chef.", "chef", "patient", "active"),  # chef作为patient
]

# 对称句对: A verb B vs B verb A (同一词在agent和patient位置)
SYMMETRIC_PAIRS = [
    ("The cat chases the dog.", "The dog chases the cat.", "cat", "dog", "chases"),
    ("The cat watches the dog.", "The dog watches the cat.", "cat", "dog", "watches"),
    ("The man helps the woman.", "The woman helps the man.", "man", "woman", "helps"),
    ("The boy follows the girl.", "The girl follows the boy.", "boy", "girl", "follows"),
    ("The dog bites the cat.", "The cat bites the dog.", "dog", "cat", "bites"),
    ("The teacher praises the student.", "The student praises the teacher.", "teacher", "student", "praises"),
    ("The doctor advises the nurse.", "The nurse advises the doctor.", "doctor", "nurse", "advises"),
    ("The writer likes the editor.", "The editor likes the writer.", "writer", "editor", "likes"),
    ("The cat sees the dog.", "The dog sees the cat.", "cat", "dog", "sees"),
    ("The man knows the woman.", "The woman knows the man.", "man", "woman", "knows"),
]


# ============================================================
# 51A: 词类控制的角色探测
# ============================================================
def exp_51a_word_class_controlled_probing(model, tokenizer, info, model_name):
    """
    核心问题: Agent探测90%+是因为"这是名词"还是"这是施事"?
    
    关键设计: 只看名词，区分agent名词 vs patient名词
    如果探测准确率仍高 → 角色确实是局部属性
    如果探测准确率降至随机 → 90%+是词类伪影
    """
    print("\n" + "="*70)
    print("51A: Word-Class Controlled Role Probing — 词类控制的角色探测")
    print("="*70)
    
    n_layers = info.n_layers
    d_model = info.d_model
    device = next(model.parameters()).device
    sample_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    
    # --- 步骤1: 收集"同词不同角色"的数据 ---
    print("\n--- Collecting same-word different-role data ---")
    
    # 用SYMMETRIC_PAIRS: A verb B 和 B verb A
    # 在第一句中A是agent，在第二句中A是patient
    word_agent_h = defaultdict(lambda: defaultdict(list))  # word -> layer -> [h_agent]
    word_patient_h = defaultdict(lambda: defaultdict(list))  # word -> layer -> [h_patient]
    
    for sent1, sent2, word1, word2, verb in SYMMETRIC_PAIRS:
        h1, tid1 = get_all_token_hidden_states(model, tokenizer, sent1, n_layers, device)
        h2, tid2 = get_all_token_hidden_states(model, tokenizer, sent2, n_layers, device)
        
        # word1在sent1中是agent，在sent2中是patient
        pos1_agent = find_token_position(tid1, tokenizer, word1)
        pos2_patient = find_token_position(tid2, tokenizer, word1)
        
        # word2在sent1中是patient，在sent2中是agent
        pos1_patient = find_token_position(tid1, tokenizer, word2)
        pos2_agent = find_token_position(tid2, tokenizer, word2)
        
        for li in sample_layers:
            if li not in h1 or li not in h2:
                continue
            # word1: agent in sent1
            for p in pos1_agent:
                if p < h1[li].shape[0]:
                    word_agent_h[word1][li].append(h1[li][p])
            # word1: patient in sent2
            for p in pos2_patient:
                if p < h2[li].shape[0]:
                    word_patient_h[word1][li].append(h2[li][p])
            # word2: patient in sent1
            for p in pos1_patient:
                if p < h1[li].shape[0]:
                    word_patient_h[word2][li].append(h1[li][p])
            # word2: agent in sent2
            for p in pos2_agent:
                if p < h2[li].shape[0]:
                    word_agent_h[word2][li].append(h2[li][p])
        
        del h1, h2
        gc.collect()
    
    # --- 步骤2: 只在名词中区分agent vs patient ---
    print("\n--- Analysis 1: Noun-Only Agent vs Patient Probing ---")
    print("  (Controlling for word class: only nouns, agent vs patient)")
    
    for li in sample_layers:
        X_agent = []
        X_patient = []
        
        # 收集所有名词在agent位置的h和patient位置的h
        for word in set(list(word_agent_h.keys()) + list(word_patient_h.keys())):
            if li in word_agent_h[word]:
                X_agent.extend(word_agent_h[word][li])
            if li in word_patient_h[word]:
                X_patient.extend(word_patient_h[word][li])
        
        if len(X_agent) < 5 or len(X_patient) < 5:
            print(f"  Layer {li:3d}: Insufficient data (N_agent={len(X_agent)}, N_patient={len(X_patient)})")
            continue
        
        X = np.array(X_agent + X_patient)
        y = np.array([1]*len(X_agent) + [0]*len(X_patient))
        
        # 随机基线
        random_acc = max(len(X_agent), len(X_patient)) / len(y)
        
        # 线性探测(Raw)
        try:
            clf = LogisticRegression(max_iter=2000, C=1.0)
            scores = cross_val_score(clf, X, y, cv=min(5, len(X_agent)), scoring='accuracy')
            raw_acc = scores.mean()
            raw_std = scores.std()
        except:
            raw_acc = random_acc
            raw_std = 0
        
        # 归一化后探测
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
        try:
            clf_norm = LogisticRegression(max_iter=2000, C=1.0)
            scores_norm = cross_val_score(clf_norm, X_norm, y, cv=min(5, len(X_agent)), scoring='accuracy')
            norm_acc = scores_norm.mean()
            norm_std = scores_norm.std()
        except:
            norm_acc = random_acc
            norm_std = 0
        
        print(f"  Layer {li:3d}: Raw acc={raw_acc:.3f}±{raw_std:.3f}, Norm acc={norm_acc:.3f}±{norm_std:.3f}, Random={random_acc:.3f} (N_agent={len(X_agent)}, N_patient={len(X_patient)})")
    
    # --- 步骤3: 同词内agent vs patient (最强控制) ---
    print("\n--- Analysis 2: Within-Word Agent vs Patient (Strongest Control) ---")
    print("  (Same word, different role: h(cat_as_agent) vs h(cat_as_patient))")
    
    for li in sample_layers:
        # 对每个有agent和patient数据的词，计算agent-patient距离
        within_word_diffs = []
        cross_word_diffs = []
        same_word_acc_list = []
        
        for word in set(list(word_agent_h.keys()) + list(word_patient_h.keys())):
            if li not in word_agent_h[word] or li not in word_patient_h[word]:
                continue
            if len(word_agent_h[word][li]) < 1 or len(word_patient_h[word][li]) < 1:
                continue
            
            h_ag = np.array(word_agent_h[word][li])
            h_pt = np.array(word_patient_h[word][li])
            
            # Agent和Patient的平均cos距离
            mean_ag = h_ag.mean(axis=0)
            mean_pt = h_pt.mean(axis=0)
            
            cos_ag_pt = np.dot(mean_ag, mean_pt) / (np.linalg.norm(mean_ag) * np.linalg.norm(mean_pt) + 1e-10)
            within_word_diffs.append(cos_ag_pt)
        
        if within_word_diffs:
            mean_cos = np.mean(within_word_diffs)
            print(f"  Layer {li:3d}: Same word agent-patient cos={mean_cos:.3f} (N_words={len(within_word_diffs)})")
        else:
            print(f"  Layer {li:3d}: No words with both agent and patient data")
    
    # --- 步骤4: 直接比较同词agent vs patient的cos相似度 ---
    print("\n--- Analysis 3: Same Word, Agent vs Patient Cosine Similarity ---")
    print("  (How much does role change the representation of the same word?)")
    
    for li in [sample_layers[0], sample_layers[len(sample_layers)//2], sample_layers[-1]]:
        for word in sorted(set(list(word_agent_h.keys()) + list(word_patient_h.keys()))):
            if li not in word_agent_h[word] or li not in word_patient_h[word]:
                continue
            if len(word_agent_h[word][li]) < 1 or len(word_patient_h[word][li]) < 1:
                continue
            
            h_ag = np.array(word_agent_h[word][li]).mean(axis=0)
            h_pt = np.array(word_patient_h[word][li]).mean(axis=0)
            cos = np.dot(h_ag, h_pt) / (np.linalg.norm(h_ag) * np.linalg.norm(h_pt) + 1e-10)
            print(f"  Layer {li:3d}, word='{word}': cos(agent,patient)={cos:.3f}")
    
    print("\n" + "="*70)
    print("51A SUMMARY: Word-Class Controlled Role Probing")
    print("="*70)


# ============================================================
# 51B: 角色向量构造
# ============================================================
def exp_51b_role_vector_construction(model, tokenizer, info, model_name):
    """
    核心问题: 如果角色是局部属性，能否提取"纯角色编码"?
    
    方法: role_vec = h(word_as_agent) - h(word_as_patient)
    
    如果role_vec一致(跨不同词) → 存在统一的"角色空间"
    如果role_vec不一致 → 角色编码与语义纠缠
    """
    print("\n" + "="*70)
    print("51B: Role Vector Construction — 角色向量构造")
    print("="*70)
    
    n_layers = info.n_layers
    d_model = info.d_model
    device = next(model.parameters()).device
    sample_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    
    # 收集数据
    print("\n--- Collecting role vector data ---")
    role_vectors = defaultdict(lambda: defaultdict(list))  # layer -> word -> [h_agent - h_patient]
    word_representations = defaultdict(lambda: defaultdict(dict))  # layer -> word -> {role: h}
    
    for sent1, sent2, word1, word2, verb in SYMMETRIC_PAIRS:
        h1, tid1 = get_all_token_hidden_states(model, tokenizer, sent1, n_layers, device)
        h2, tid2 = get_all_token_hidden_states(model, tokenizer, sent2, n_layers, device)
        
        for li in sample_layers:
            if li not in h1 or li not in h2:
                continue
            
            # word1: agent in sent1, patient in sent2
            pos1 = find_token_position(tid1, tokenizer, word1)
            pos2 = find_token_position(tid2, tokenizer, word1)
            
            if pos1 and pos2 and pos1[0] < h1[li].shape[0] and pos2[0] < h2[li].shape[0]:
                h_ag = h1[li][pos1[0]]
                h_pt = h2[li][pos2[0]]
                role_vec = h_ag - h_pt
                role_vectors[li][word1].append(role_vec)
                word_representations[li][word1]['agent'] = h_ag
                word_representations[li][word1]['patient'] = h_pt
            
            # word2: patient in sent1, agent in sent2
            pos1 = find_token_position(tid1, tokenizer, word2)
            pos2 = find_token_position(tid2, tokenizer, word2)
            
            if pos1 and pos2 and pos1[0] < h1[li].shape[0] and pos2[0] < h2[li].shape[0]:
                h_ag = h2[li][pos2[0]]
                h_pt = h1[li][pos1[0]]
                role_vec = h_ag - h_pt
                role_vectors[li][word2].append(role_vec)
                word_representations[li][word2]['agent'] = h_ag
                word_representations[li][word2]['patient'] = h_pt
        
        del h1, h2
        gc.collect()
    
    # --- Analysis 1: 角色向量跨词一致性 ---
    print("\n--- Analysis 1: Role Vector Cross-Word Consistency ---")
    print("  (Is the agent-patient direction similar across different words?)")
    
    for li in sample_layers:
        vecs = []
        words_with_vecs = []
        for word, vec_list in role_vectors[li].items():
            if vec_list:
                vecs.append(np.mean(vec_list, axis=0))
                words_with_vecs.append(word)
        
        if len(vecs) < 2:
            print(f"  Layer {li:3d}: Insufficient words with role vectors")
            continue
        
        vecs = np.array(vecs)
        # 计算两两cos相似度
        cos_matrix = np.zeros((len(vecs), len(vecs)))
        for i in range(len(vecs)):
            for j in range(i+1, len(vecs)):
                ni = np.linalg.norm(vecs[i])
                nj = np.linalg.norm(vecs[j])
                if ni > 1e-10 and nj > 1e-10:
                    cos_matrix[i][j] = np.dot(vecs[i], vecs[j]) / (ni * nj)
                    cos_matrix[j][i] = cos_matrix[i][j]
        
        upper_tri = cos_matrix[np.triu_indices(len(vecs), k=1)]
        mean_cos = upper_tri.mean()
        std_cos = upper_tri.std()
        
        # 角色向量的模长
        norms = [np.linalg.norm(v) for v in vecs]
        mean_norm = np.mean(norms)
        
        print(f"  Layer {li:3d}: Cross-word role_vec cos={mean_cos:.3f}±{std_cos:.3f}, ||role_vec||={mean_norm:.1f}, N_words={len(words_with_vecs)}")
    
    # --- Analysis 2: 角色向量PCA ---
    print("\n--- Analysis 2: Role Vector PCA Dimensionality ---")
    
    for li in [sample_layers[len(sample_layers)//2], sample_layers[-1]]:
        all_vecs = []
        for word, vec_list in role_vectors[li].items():
            all_vecs.extend(vec_list)
        
        if len(all_vecs) < 5:
            continue
        
        all_vecs = np.array(all_vecs)
        pca = PCA()
        pca.fit(all_vecs)
        
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        dim90 = np.searchsorted(cumvar, 0.9) + 1
        top1 = pca.explained_variance_ratio_[0]
        top5 = pca.explained_variance_ratio_[:5].sum()
        
        print(f"  Layer {li:3d}: dim90={dim90}, top1={top1:.3f}, top5={top5:.3f}")
    
    # --- Analysis 3: 角色向量的方向 vs 语义方向 ---
    print("\n--- Analysis 3: Role Direction vs Semantic Direction ---")
    print("  (Is the agent-patient direction orthogonal to the word identity direction?)")
    
    for li in [sample_layers[len(sample_layers)//2], sample_layers[-1]]:
        angles = []
        for word, reps in word_representations[li].items():
            if 'agent' not in reps or 'patient' not in reps:
                continue
            h_ag = reps['agent']
            h_pt = reps['patient']
            role_vec = h_ag - h_pt
            semantic_vec = (h_ag + h_pt) / 2  # 平均表示 ≈ 语义核心
            
            ni = np.linalg.norm(role_vec)
            nj = np.linalg.norm(semantic_vec)
            if ni > 1e-10 and nj > 1e-10:
                cos = np.dot(role_vec, semantic_vec) / (ni * nj)
                angles.append(cos)
        
        if angles:
            print(f"  Layer {li:3d}: cos(role_vec, semantic_vec)={np.mean(angles):.3f}±{np.std(angles):.3f} (0=orthogonal, 1=parallel)")
    
    # --- Analysis 4: 角色向量能否预测其他词的角色 ---
    print("\n--- Analysis 4: Role Vector Transfer ---")
    print("  (Can role_vec from word A predict agent/patient for word B?)")
    
    for li in [sample_layers[len(sample_layers)//2], sample_layers[-1]]:
        words_list = [w for w in role_vectors[li] if role_vectors[li][w]]
        if len(words_list) < 4:
            continue
        
        # 计算平均角色向量(方向)
        all_role_vecs = []
        for w in words_list:
            all_role_vecs.append(np.mean(role_vectors[li][w], axis=0))
        mean_role_dir = np.mean(all_role_vecs, axis=0)
        mean_role_dir = mean_role_dir / (np.linalg.norm(mean_role_dir) + 1e-10)
        
        # 对每个词，检查: h_agent投影到role_dir > h_patient投影到role_dir?
        correct = 0
        total = 0
        for word, reps in word_representations[li].items():
            if 'agent' not in reps or 'patient' not in reps:
                continue
            proj_ag = np.dot(reps['agent'], mean_role_dir)
            proj_pt = np.dot(reps['patient'], mean_role_dir)
            if proj_ag > proj_pt:
                correct += 1
            total += 1
        
        acc = correct / total if total > 0 else 0
        print(f"  Layer {li:3d}: Role direction transfer acc={acc:.3f} (N={total}, random=0.5)")
    
    print("\n" + "="*70)
    print("51B SUMMARY: Role Vector Construction")
    print("="*70)


# ============================================================
# 51C: 角色向量代数
# ============================================================
def exp_51c_role_vector_algebra(model, tokenizer, info, model_name):
    """
    核心问题: 角色向量有什么代数性质?
    
    1. agent_vec + patient_vec → 无意义/中性?
    2. 交换角色: h(agent→patient) + role_vec = h(patient→agent)?
    3. 角色向量与位置编码的关系
    """
    print("\n" + "="*70)
    print("51C: Role Vector Algebra — 角色向量代数")
    print("="*70)
    
    n_layers = info.n_layers
    d_model = info.d_model
    device = next(model.parameters()).device
    sample_layers = [0, n_layers//2, n_layers-1]
    
    # 收集数据
    print("\n--- Collecting data ---")
    all_data = {}
    for sent1, sent2, word1, word2, verb in SYMMETRIC_PAIRS[:8]:
        h1, tid1 = get_all_token_hidden_states(model, tokenizer, sent1, n_layers, device)
        h2, tid2 = get_all_token_hidden_states(model, tokenizer, sent2, n_layers, device)
        all_data[(sent1, sent2)] = (h1, tid1, h2, tid2)
        gc.collect()
    
    # --- Analysis 1: 角色向量加法 ---
    print("\n--- Analysis 1: Role Vector Addition ---")
    print("  If role_vec = h_agent - h_patient, does adding role_vec swap roles?")
    
    for li in sample_layers:
        swap_success = []
        
        for sent1, sent2, word1, word2, verb in SYMMETRIC_PAIRS[:8]:
            if (sent1, sent2) not in all_data:
                continue
            h1, tid1, h2, tid2 = all_data[(sent1, sent2)]
            
            if li not in h1 or li not in h2:
                continue
            
            # word1在sent1是agent，sent2是patient
            pos1_ag = find_token_position(tid1, tokenizer, word1)
            pos2_pt = find_token_position(tid2, tokenizer, word1)
            
            if not pos1_ag or not pos2_pt:
                continue
            if pos1_ag[0] >= h1[li].shape[0] or pos2_pt[0] >= h2[li].shape[0]:
                continue
            
            h_agent = h1[li][pos1_ag[0]]
            h_patient = h2[li][pos2_pt[0]]
            role_vec = h_agent - h_patient
            
            # 验证: h_patient + role_vec ≈ h_agent?
            h_predicted_agent = h_patient + role_vec
            cos_pred_actual = np.dot(h_predicted_agent, h_agent) / (np.linalg.norm(h_predicted_agent) * np.linalg.norm(h_agent) + 1e-10)
            cos_pred_patient = np.dot(h_predicted_agent, h_patient) / (np.linalg.norm(h_predicted_agent) * np.linalg.norm(h_patient) + 1e-10)
            
            swap_success.append({
                'word': word1,
                'cos_with_agent': cos_pred_actual,
                'cos_with_patient': cos_pred_patient,
                'prefer_agent': cos_pred_actual > cos_pred_patient
            })
        
        if swap_success:
            mean_cos_ag = np.mean([s['cos_with_agent'] for s in swap_success])
            mean_cos_pt = np.mean([s['cos_with_patient'] for s in swap_success])
            prefer_rate = np.mean([s['prefer_agent'] for s in swap_success])
            print(f"  Layer {li:3d}: h_pt + role_vec → cos(agent)={mean_cos_ag:.3f}, cos(patient)={mean_cos_pt:.3f}, prefer_agent={prefer_rate:.3f}")
    
    # --- Analysis 2: 角色向量交换(跨词) ---
    print("\n--- Analysis 2: Cross-Word Role Vector Transfer ---")
    print("  Does role_vec from word A work for word B?")
    
    for li in sample_layers:
        cross_word_results = []
        words_data = []
        
        for sent1, sent2, word1, word2, verb in SYMMETRIC_PAIRS[:8]:
            if (sent1, sent2) not in all_data:
                continue
            h1, tid1, h2, tid2 = all_data[(sent1, sent2)]
            
            if li not in h1 or li not in h2:
                continue
            
            for word in [word1, word2]:
                pos1 = find_token_position(tid1, tokenizer, word)
                pos2 = find_token_position(tid2, tokenizer, word)
                
                if not pos1 or not pos2:
                    continue
                if pos1[0] >= h1[li].shape[0] or pos2[0] >= h2[li].shape[0]:
                    continue
                
                h_ag = h1[li][pos1[0]] if word == word1 else h2[li][pos2[0]]
                h_pt = h2[li][pos2[0]] if word == word1 else h1[li][pos1[0]]
                
                # 确保agent/patient正确
                # word1 in sent1 is agent, in sent2 is patient
                # word2 in sent1 is patient, in sent2 is agent
                if word == word1:
                    h_ag = h1[li][pos1[0]]
                    h_pt = h2[li][pos2[0]]
                else:
                    h_ag = h2[li][pos2[0]]
                    h_pt = h1[li][pos1[0]]
                
                words_data.append({
                    'word': word,
                    'h_agent': h_ag,
                    'h_patient': h_pt,
                    'role_vec': h_ag - h_pt
                })
        
        if len(words_data) < 2:
            continue
        
        # 用词A的role_vec修改词B的表示
        for i, wd_a in enumerate(words_data):
            for j, wd_b in enumerate(words_data):
                if i == j:
                    continue
                # h_b_patient + role_vec_a ≈ h_b_agent?
                h_modified = wd_b['h_patient'] + wd_a['role_vec']
                cos_with_ag = np.dot(h_modified, wd_b['h_agent']) / (np.linalg.norm(h_modified) * np.linalg.norm(wd_b['h_agent']) + 1e-10)
                cos_with_pt = np.dot(h_modified, wd_b['h_patient']) / (np.linalg.norm(h_modified) * np.linalg.norm(wd_b['h_patient']) + 1e-10)
                cross_word_results.append(cos_with_ag > cos_with_pt)
        
        if cross_word_results:
            transfer_rate = np.mean(cross_word_results)
            print(f"  Layer {li:3d}: Cross-word role transfer rate={transfer_rate:.3f} (random=0.5, N={len(cross_word_results)})")
    
    # --- Analysis 3: 角色向量vs位置的关系 ---
    print("\n--- Analysis 3: Role Vector vs Position ---")
    print("  (Is role encoding just a position encoding effect?)")
    
    # 在主动句中agent在位置2-3，patient在位置5-7
    # 在被动句中agent在by之后，patient在位置2-3
    # 如果角色=位置，则同一位置的不同角色词应该相似
    
    for li in sample_layers:
        pos2_reps = []  # 位置2的表示（主动句agent，被动句patient）
        pos5_reps = []  # 位置5-6的表示（主动句patient，被动句的其他）
        
        for sent1, sent2, word1, word2, verb in SYMMETRIC_PAIRS[:8]:
            if (sent1, sent2) not in all_data:
                continue
            h1, tid1, h2, tid2 = all_data[(sent1, sent2)]
            
            if li not in h1 or li not in h2:
                continue
            
            # 主动句: agent大约在pos 2-3
            # 被动句: patient大约在pos 2-3 (主语位置)
            seq1_len = h1[li].shape[0]
            seq2_len = h2[li].shape[0]
            
            if seq1_len > 3:
                pos2_reps.append(('active_agent', h1[li][2]))  # agent位置
            if seq2_len > 3:
                pos2_reps.append(('passive_patient', h2[li][2]))  # patient位置
        
        if len(pos2_reps) > 4:
            active_reps = [r[1] for r in pos2_reps if r[0] == 'active_agent']
            passive_reps = [r[1] for r in pos2_reps if r[0] == 'passive_patient']
            
            if active_reps and passive_reps:
                # 同位置不同角色的cos
                mean_active = np.mean(active_reps, axis=0)
                mean_passive = np.mean(passive_reps, axis=0)
                cos = np.dot(mean_active, mean_passive) / (np.linalg.norm(mean_active) * np.linalg.norm(mean_passive) + 1e-10)
                print(f"  Layer {li:3d}: Same position(pos2), diff role cos={cos:.3f} (active_agent vs passive_patient)")
    
    print("\n" + "="*70)
    print("51C SUMMARY: Role Vector Algebra")
    print("="*70)


# ============================================================
# 51D: 主动/被动变换中的角色向量变化
# ============================================================
def exp_51d_role_vector_voice_transform(model, tokenizer, info, model_name):
    """
    核心问题: 主动→被动变换如何影响角色编码?
    
    Phase 50发现: Agent在active/passive中cos=0.59-0.95
    → 角色编码确实被语态改变
    → 量化这个改变的结构
    """
    print("\n" + "="*70)
    print("51D: Role Vector Under Voice Transform — 语态变换下的角色向量")
    print("="*70)
    
    n_layers = info.n_layers
    d_model = info.d_model
    device = next(model.parameters()).device
    sample_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    
    # 主动-被动对
    active_passive_pairs = [
        ("The cat chases the dog.", "The dog is chased by the cat.", "cat", "dog"),
        ("The man reads the book.", "The book is read by the man.", "man", "book"),
        ("The boy throws the ball.", "The ball is thrown by the boy.", "boy", "ball"),
        ("The fox hunts the rabbit.", "The rabbit is hunted by the fox.", "fox", "rabbit"),
        ("The bear eats the honey.", "The honey is eaten by the bear.", "bear", "honey"),
        ("The dog bites the bone.", "The bone is bitten by the dog.", "dog", "bone"),
        ("The teacher writes the letter.", "The letter is written by the teacher.", "teacher", "letter"),
        ("The chef cooks the meal.", "The meal is cooked by the chef.", "chef", "meal"),
    ]
    
    print("\n--- Collecting data ---")
    all_data = {}
    for active, passive, agent, patient in active_passive_pairs:
        h_act, tid_act = get_all_token_hidden_states(model, tokenizer, active, n_layers, device)
        h_pas, tid_pas = get_all_token_hidden_states(model, tokenizer, passive, n_layers, device)
        all_data[(active, passive)] = (h_act, tid_act, h_pas, tid_pas)
        gc.collect()
    
    # --- Analysis 1: Agent在active vs passive中的表示 ---
    print("\n--- Analysis 1: Agent Representation Active vs Passive ---")
    
    for li in sample_layers:
        agent_active_cos = []
        patient_active_cos = []
        
        for active, passive, agent, patient in active_passive_pairs:
            if (active, passive) not in all_data:
                continue
            h_act, tid_act, h_pas, tid_pas = all_data[(active, passive)]
            
            if li not in h_act or li not in h_pas:
                continue
            
            # Agent在active中的位置
            pos_ag_act = find_token_position(tid_act, tokenizer, agent)
            # Agent在passive中的位置 (by the agent)
            pos_ag_pas = find_token_position(tid_pas, tokenizer, agent)
            # Patient在active中的位置
            pos_pt_act = find_token_position(tid_act, tokenizer, patient)
            # Patient在passive中的位置 (语法主语)
            pos_pt_pas = find_token_position(tid_pas, tokenizer, patient)
            
            if not pos_ag_act or not pos_ag_pas:
                continue
            if pos_ag_act[0] >= h_act[li].shape[0] or pos_ag_pas[0] >= h_pas[li].shape[0]:
                continue
            
            # Agent: active位置 vs passive位置
            h_ag_act = h_act[li][pos_ag_act[0]]
            h_ag_pas = h_pas[li][pos_ag_pas[0]]
            cos_ag = np.dot(h_ag_act, h_ag_pas) / (np.linalg.norm(h_ag_act) * np.linalg.norm(h_ag_pas) + 1e-10)
            agent_active_cos.append(cos_ag)
            
            # Patient: active位置 vs passive位置
            if pos_pt_act and pos_pt_pas and pos_pt_act[0] < h_act[li].shape[0] and pos_pt_pas[0] < h_pas[li].shape[0]:
                h_pt_act = h_act[li][pos_pt_act[0]]
                h_pt_pas = h_pas[li][pos_pt_pas[0]]
                cos_pt = np.dot(h_pt_act, h_pt_pas) / (np.linalg.norm(h_pt_act) * np.linalg.norm(h_pt_pas) + 1e-10)
                patient_active_cos.append(cos_pt)
        
        agent_str = f"agent: cos={np.mean(agent_active_cos):.3f}±{np.std(agent_active_cos):.3f}" if agent_active_cos else "agent: N/A"
        patient_str = f"patient: cos={np.mean(patient_active_cos):.3f}±{np.std(patient_active_cos):.3f}" if patient_active_cos else "patient: N/A"
        print(f"  Layer {li:3d}: {agent_str}, {patient_str}")
    
    # --- Analysis 2: 角色向量在active vs passive中的比较 ---
    print("\n--- Analysis 2: Role Vector Active vs Passive ---")
    print("  role_vec_active = h(agent, active) - h(patient, active)")
    print("  role_vec_passive = h(agent, passive) - h(patient, passive)")
    
    for li in sample_layers:
        role_vec_coss = []
        
        for active, passive, agent, patient in active_passive_pairs:
            if (active, passive) not in all_data:
                continue
            h_act, tid_act, h_pas, tid_pas = all_data[(active, passive)]
            
            if li not in h_act or li not in h_pas:
                continue
            
            pos_ag_act = find_token_position(tid_act, tokenizer, agent)
            pos_pt_act = find_token_position(tid_act, tokenizer, patient)
            pos_ag_pas = find_token_position(tid_pas, tokenizer, agent)
            pos_pt_pas = find_token_position(tid_pas, tokenizer, patient)
            
            if not all([pos_ag_act, pos_pt_act, pos_ag_pas, pos_pt_pas]):
                continue
            if pos_ag_act[0] >= h_act[li].shape[0] or pos_pt_act[0] >= h_act[li].shape[0]:
                continue
            if pos_ag_pas[0] >= h_pas[li].shape[0] or pos_pt_pas[0] >= h_pas[li].shape[0]:
                continue
            
            # Active: agent在主语位置，patient在宾语位置
            rv_act = h_act[li][pos_ag_act[0]] - h_act[li][pos_pt_act[0]]
            # Passive: agent在by后，patient在主语位置
            rv_pas = h_pas[li][pos_ag_pas[0]] - h_pas[li][pos_pt_pas[0]]
            
            n1 = np.linalg.norm(rv_act)
            n2 = np.linalg.norm(rv_pas)
            if n1 > 1e-10 and n2 > 1e-10:
                cos = np.dot(rv_act, rv_pas) / (n1 * n2)
                role_vec_coss.append(cos)
        
        if role_vec_coss:
            print(f"  Layer {li:3d}: cos(role_vec_active, role_vec_passive)={np.mean(role_vec_coss):.3f}±{np.std(role_vec_coss):.3f} (N={len(role_vec_coss)})")
        else:
            print(f"  Layer {li:3d}: Insufficient data")
    
    # --- Analysis 3: 语法主语位置 vs 语义角色 ---
    print("\n--- Analysis 3: Syntactic Subject Position vs Semantic Role ---")
    print("  (In passive: patient is syntactic subject, agent is in by-phrase)")
    print("  (Does position determine representation more than semantic role?)")
    
    for li in sample_layers:
        # 同一位置(position 2), 不同语义角色
        # active: pos2是agent(语义+语法主语)
        # passive: pos2是patient(语法主语但语义patient)
        same_pos_diff_role = []
        
        for active, passive, agent, patient in active_passive_pairs:
            if (active, passive) not in all_data:
                continue
            h_act, tid_act, h_pas, tid_pas = all_data[(active, passive)]
            
            if li not in h_act or li not in h_pas:
                continue
            
            seq_act = h_act[li].shape[0]
            seq_pas = h_pas[li].shape[0]
            
            # 主动句pos2 = agent, 被动句pos2 = patient
            if seq_act > 3 and seq_pas > 3:
                h_pos2_active = h_act[li][2]  # agent
                h_pos2_passive = h_pas[li][2]  # patient(语法主语)
                cos = np.dot(h_pos2_active, h_pos2_passive) / (np.linalg.norm(h_pos2_active) * np.linalg.norm(h_pos2_passive) + 1e-10)
                same_pos_diff_role.append(cos)
        
        if same_pos_diff_role:
            print(f"  Layer {li:3d}: Same pos(2), diff semantic role cos={np.mean(same_pos_diff_role):.3f} (active_agent vs passive_patient)")
    
    print("\n" + "="*70)
    print("51D SUMMARY: Role Vector Under Voice Transform")
    print("="*70)


# ============================================================
# 51E: 语义-角色正交性
# ============================================================
def exp_51e_semantic_role_orthogonality(model, tokenizer, info, model_name):
    """
    核心问题: 语义维度和角色维度是否正交?
    
    如果 h = semantic_component + role_component
    且 semantic ⊥ role
    则可以通过投影分离
    
    验证方法:
    1. 去语义后角色是否可读
    2. 去角色后语义是否可读
    3. 角色子空间和语义子空间的cos
    """
    print("\n" + "="*70)
    print("51E: Semantic-Role Orthogonality — 语义-角色正交性")
    print("="*70)
    
    n_layers = info.n_layers
    d_model = info.d_model
    device = next(model.parameters()).device
    sample_layers = [0, n_layers//2, n_layers-1]
    
    # 收集对称句的数据
    print("\n--- Collecting data ---")
    all_data = {}
    for sent1, sent2, word1, word2, verb in SYMMETRIC_PAIRS[:8]:
        h1, tid1 = get_all_token_hidden_states(model, tokenizer, sent1, n_layers, device)
        h2, tid2 = get_all_token_hidden_states(model, tokenizer, sent2, n_layers, device)
        all_data[(sent1, sent2)] = (h1, tid1, h2, tid2)
        gc.collect()
    
    # --- Analysis 1: 用PCA分离语义和角色 ---
    print("\n--- Analysis 1: PCA-Based Semantic-Role Separation ---")
    
    for li in sample_layers:
        # 收集所有名词token的h，标记(word_id, role)
        X = []
        word_labels = []
        role_labels = []  # 1=agent, 0=patient
        
        word_to_id = {}
        next_id = 0
        
        for sent1, sent2, word1, word2, verb in SYMMETRIC_PAIRS[:8]:
            if (sent1, sent2) not in all_data:
                continue
            h1, tid1, h2, tid2 = all_data[(sent1, sent2)]
            
            if li not in h1 or li not in h2:
                continue
            
            for word, h_source, tid_source, is_agent_in_source in [
                (word1, h1, tid1, True),   # word1 in sent1 is agent
                (word1, h2, tid2, False),  # word1 in sent2 is patient
                (word2, h1, tid1, False),  # word2 in sent1 is patient
                (word2, h2, tid2, True),   # word2 in sent2 is agent
            ]:
                pos = find_token_position(tid_source, tokenizer, word)
                if not pos or pos[0] >= h_source[li].shape[0]:
                    continue
                
                if word not in word_to_id:
                    word_to_id[word] = next_id
                    next_id += 1
                
                X.append(h_source[li][pos[0]])
                word_labels.append(word_to_id[word])
                role_labels.append(1 if is_agent_in_source else 0)
        
        if len(X) < 10:
            continue
        
        X = np.array(X)
        word_labels = np.array(word_labels)
        role_labels = np.array(role_labels)
        
        # PCA降维
        pca = PCA(n_components=min(50, X.shape[0]-1, X.shape[1]))
        X_pca = pca.fit_transform(X)
        
        # 前10个PCA方向能否预测word identity?
        # 后面的PCA方向能否预测role?
        n_semantic = min(10, X_pca.shape[1])
        n_role = min(10, X_pca.shape[1] - n_semantic)
        
        if X_pca.shape[1] > n_semantic and len(set(word_labels)) > 2:
            # 语义预测: 用前n_semantic个PC
            try:
                clf_word = LogisticRegression(max_iter=2000, C=1.0, multi_class='ovr')
                scores_word = cross_val_score(clf_word, X_pca[:, :n_semantic], word_labels, 
                                              cv=min(3, len(set(word_labels))), scoring='accuracy')
                word_acc_top = scores_word.mean()
            except:
                word_acc_top = -1
            
            # 角色预测: 用后n_role个PC
            try:
                clf_role = LogisticRegression(max_iter=2000, C=1.0)
                scores_role = cross_val_score(clf_role, X_pca[:, n_semantic:n_semantic+n_role], role_labels, 
                                              cv=min(3, min(sum(role_labels), len(role_labels)-sum(role_labels))), scoring='accuracy')
                role_acc_bottom = scores_role.mean()
            except:
                role_acc_bottom = -1
            
            # 对比: 前10个PC预测role vs 后10个PC预测role
            try:
                clf_role_top = LogisticRegression(max_iter=2000, C=1.0)
                scores_role_top = cross_val_score(clf_role_top, X_pca[:, :n_semantic], role_labels, 
                                                  cv=min(3, min(sum(role_labels), len(role_labels)-sum(role_labels))), scoring='accuracy')
                role_acc_top = scores_role_top.mean()
            except:
                role_acc_top = -1
            
            print(f"  Layer {li:3d}: PC1-{n_semantic}: word_acc={word_acc_top:.3f}, role_acc={role_acc_top:.3f}")
            print(f"           PC{n_semantic+1}-{n_semantic+n_role}: role_acc={role_acc_bottom:.3f}")
    
    # --- Analysis 2: 投影分离验证 ---
    print("\n--- Analysis 2: Projection-Based Separation ---")
    print("  (Remove word-identity directions, then test role probing)")
    
    for li in sample_layers:
        X = []
        word_labels = []
        role_labels = []
        word_to_id = {}
        next_id = 0
        
        for sent1, sent2, word1, word2, verb in SYMMETRIC_PAIRS[:8]:
            if (sent1, sent2) not in all_data:
                continue
            h1, tid1, h2, tid2 = all_data[(sent1, sent2)]
            
            if li not in h1 or li not in h2:
                continue
            
            for word, h_source, tid_source, is_agent in [
                (word1, h1, tid1, True), (word1, h2, tid2, False),
                (word2, h1, tid1, False), (word2, h2, tid2, True),
            ]:
                pos = find_token_position(tid_source, tokenizer, word)
                if not pos or pos[0] >= h_source[li].shape[0]:
                    continue
                if word not in word_to_id:
                    word_to_id[word] = next_id
                    next_id += 1
                X.append(h_source[li][pos[0]])
                word_labels.append(word_to_id[word])
                role_labels.append(1 if is_agent else 0)
        
        if len(X) < 10:
            continue
        
        X = np.array(X)
        role_labels = np.array(role_labels)
        word_labels = np.array(word_labels)
        
        # 方法1: 用LDA找word-identity方向，然后投影掉
        if len(set(word_labels)) >= 3:
            try:
                lda_word = LinearDiscriminantAnalysis()
                lda_word.fit(X, word_labels)
                # 投影到word-identity子空间
                word_subspace = lda_word.scalings_[:, :min(5, lda_word.scalings_.shape[1])] if hasattr(lda_word, 'scalings_') else None
                
                if word_subspace is not None:
                    # 投影掉word子空间
                    proj = word_subspace @ word_subspace.T
                    X_no_word = X - (X @ proj) @ np.eye(proj.shape[0]) if False else None
                    
                    # 简化: 用PCA方法
                    # 找到word-identity的主方向
                    pca_full = PCA(n_components=min(20, X.shape[1]))
                    X_pca = pca_full.fit_transform(X)
                    
                    # 用前k个PC重构，去掉(认为是语义)
                    # 用残差做角色探测
                    for k in [5, 10]:
                        X_reconstructed = X_pca[:, :k] @ pca_full.components_[:k] + pca_full.mean_
                        X_residual = X - X_reconstructed
                        
                        try:
                            clf = LogisticRegression(max_iter=2000, C=1.0)
                            scores = cross_val_score(clf, X_residual, role_labels, 
                                                      cv=min(3, min(sum(role_labels), len(role_labels)-sum(role_labels))), scoring='accuracy')
                            residual_role_acc = scores.mean()
                        except:
                            residual_role_acc = -1
                        
                        random_baseline = max(sum(role_labels), len(role_labels)-sum(role_labels)) / len(role_labels)
                        print(f"  Layer {li:3d}: After removing top-{k} PCs, role acc={residual_role_acc:.3f} (random={random_baseline:.3f})")
            except:
                print(f"  Layer {li:3d}: LDA/PCA separation failed")
    
    # --- Analysis 3: 角色子空间和语义子空间的角度 ---
    print("\n--- Analysis 3: Subspace Angle Between Role and Semantic ---")
    
    for li in sample_layers:
        X = []
        word_labels = []
        role_labels = []
        word_to_id = {}
        next_id = 0
        
        for sent1, sent2, word1, word2, verb in SYMMETRIC_PAIRS[:8]:
            if (sent1, sent2) not in all_data:
                continue
            h1, tid1, h2, tid2 = all_data[(sent1, sent2)]
            
            if li not in h1 or li not in h2:
                continue
            
            for word, h_source, tid_source, is_agent in [
                (word1, h1, tid1, True), (word1, h2, tid2, False),
                (word2, h1, tid1, False), (word2, h2, tid2, True),
            ]:
                pos = find_token_position(tid_source, tokenizer, word)
                if not pos or pos[0] >= h_source[li].shape[0]:
                    continue
                if word not in word_to_id:
                    word_to_id[word] = next_id
                    next_id += 1
                X.append(h_source[li][pos[0]])
                word_labels.append(word_to_id[word])
                role_labels.append(1 if is_agent else 0)
        
        if len(X) < 10:
            continue
        
        X = np.array(X)
        role_labels = np.array(role_labels)
        word_labels = np.array(word_labels)
        
        # 用LDA找role方向和word方向
        try:
            lda_role = LinearDiscriminantAnalysis()
            lda_role.fit(X, role_labels)
            role_dir = lda_role.scalings_[:, 0] if hasattr(lda_role, 'scalings_') else None
        except:
            role_dir = None
        
        try:
            lda_word = LinearDiscriminantAnalysis()
            lda_word.fit(X, word_labels)
            word_dirs = lda_word.scalings_[:, :min(3, lda_word.scalings_.shape[1])] if hasattr(lda_word, 'scalings_') else None
        except:
            word_dirs = None
        
        if role_dir is not None and word_dirs is not None:
            # 计算role方向和word子空间的角度
            role_dir = role_dir / (np.linalg.norm(role_dir) + 1e-10)
            
            # 投影role_dir到word子空间
            proj = word_dirs @ np.linalg.inv(word_dirs.T @ word_dirs) @ word_dirs.T
            role_proj = proj @ role_dir
            cos_role_with_word = np.linalg.norm(role_proj)  # 如果=1则role在word子空间内
            
            print(f"  Layer {li:3d}: Role dir alignment with word subspace: {cos_role_with_word:.3f} (0=orthogonal, 1=in subspace)")
        else:
            print(f"  Layer {li:3d}: Could not compute subspace angle")
    
    print("\n" + "="*70)
    print("51E SUMMARY: Semantic-Role Orthogonality")
    print("="*70)


# ============================================================
# 主函数
# ============================================================
EXPERIMENTS = {
    1: ("51A: Word-Class Controlled Role Probing", exp_51a_word_class_controlled_probing),
    2: ("51B: Role Vector Construction", exp_51b_role_vector_construction),
    3: ("51C: Role Vector Algebra", exp_51c_role_vector_algebra),
    4: ("51D: Role Vector Voice Transform", exp_51d_role_vector_voice_transform),
    5: ("51E: Semantic-Role Orthogonality", exp_51e_semantic_role_orthogonality),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, 
                        choices=['deepseek7b', 'glm4', 'qwen3'],
                        help='Model to test')
    parser.add_argument('--exp', type=int, required=True,
                        choices=list(EXPERIMENTS.keys()),
                        help='Experiment number (1-5)')
    args = parser.parse_args()
    
    model_name = args.model
    exp_num = args.exp
    
    print(f"\n{'='*70}")
    print(f"Phase 51: Semantic-Role Separation — 语义-角色分离")
    print(f"Model: {model_name}")
    print(f"{'='*70}")
    
    # 加载模型
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    
    # 运行实验
    exp_name, exp_fn = EXPERIMENTS[exp_num]
    exp_fn(model, tokenizer, info, model_name)
    
    # 释放模型
    release_model(model)
    
    print(f"\n{'='*70}")
    print(f"Phase 51 COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
