"""
CCL-K(252): 正四面体编码的因果验证与语义解耦
==================================================
基于CCL-J的重大发现: 4个语法角色方向构成近似正四面体

Phase 8关键验证实验:
  Exp1 (8D): ★★★★★ 随机标签对照实验
    → 核心问题: 正四面体是探针的artifact, 还是hidden state的真实结构?
    → 方法: 用真实hidden states + 随机标签训练探针
    → 预测: 如果正四面体是真实结构, 随机标签不应产生正四面体
    
  Exp2 (8D-ext): ★★★★ 随机向量对照实验
    → 核心问题: LogisticRegression本身是否偏好正四面体?
    → 方法: 用随机高斯向量 + 真实标签训练探针
    → 预测: 如果LR偏好正四面体, 随机向量也会产生正四面体
    
  Exp3 (8B): ★★★★★ 语义-语法解耦操控
    → 核心问题: 探针方向中的"纯语法"成分是什么?
    → 方法: 将探针方向分解为 W_U平行分量 + W_U正交分量
    → W_U正交分量 = 只影响语法分类, 不影响token选择的"纯语法"方向
    → 如果找到: 实现语义保持的语法角色操控!
    
  Exp4 (8A): ★★★★ 正四面体对称性验证
    → 核心问题: 4个方向的对称性有多强?
    → 方法: 测量各角色的分类margin, 类内方差, 置换等变性
    → 如果对称: 分类margin相等, 置换后探针同样准确
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
from scipy.sparse.linalg import svds

from model_utils import load_model, get_layers, get_model_info, release_model, safe_decode, get_W_U


# ===== 数据集(复用CCL-J) =====
EXTENDED_DATA = {
    "nsubj": {
        "sentences": [
            "The king ruled the kingdom",
            "The doctor treated the patient",
            "The artist painted the portrait",
            "The soldier defended the castle",
            "The cat sat on the mat",
            "The dog ran through the park",
            "The bird sang a beautiful song",
            "The child played with the toys",
            "The student read the textbook",
            "The teacher explained the lesson",
            "The woman drove the car",
            "The man fixed the roof",
            "The girl sang a song",
            "The boy kicked the ball",
            "The president signed the bill",
            "The chef cooked the meal",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier",
            "cat", "dog", "bird", "child",
            "student", "teacher", "woman", "man",
            "girl", "boy", "president", "chef",
        ],
    },
    "dobj": {
        "sentences": [
            "They crowned the king yesterday",
            "She visited the doctor recently",
            "He admired the artist greatly",
            "We honored the soldier today",
            "She chased the cat away",
            "He found the dog outside",
            "They watched the bird closely",
            "We helped the child today",
            "I praised the student loudly",
            "You thanked the teacher warmly",
            "The police arrested the man quickly",
            "The company hired the woman recently",
            "The coach trained the girl daily",
            "The teacher praised the boy warmly",
            "The nation elected the president fairly",
            "The customer tipped the chef generously",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier",
            "cat", "dog", "bird", "child",
            "student", "teacher", "man", "woman",
            "girl", "boy", "president", "chef",
        ],
    },
    "amod": {
        "sentences": [
            "The brave king fought hard",
            "The kind doctor helped many",
            "The creative artist worked well",
            "The strong soldier marched far",
            "The beautiful cat sat quietly",
            "The large dog ran swiftly",
            "The small bird sang softly",
            "The young child played happily",
            "The bright student read carefully",
            "The wise teacher explained clearly",
            "The old woman walked slowly",
            "The tall man stood quietly",
            "The little girl smiled sweetly",
            "The smart boy answered quickly",
            "The powerful president decided firmly",
            "The skilled chef cooked perfectly",
        ],
        "target_words": [
            "brave", "kind", "creative", "strong",
            "beautiful", "large", "small", "young",
            "bright", "wise", "old", "tall",
            "little", "smart", "powerful", "skilled",
        ],
    },
    "advmod": {
        "sentences": [
            "The king ruled wisely forever",
            "The doctor worked carefully always",
            "The artist painted beautifully daily",
            "The soldier fought bravely there",
            "The cat ran quickly home",
            "The dog barked loudly today",
            "The bird sang softly outside",
            "The child played happily inside",
            "The student read carefully alone",
            "The teacher spoke clearly again",
            "The woman drove slowly home",
            "The man spoke quietly now",
            "The girl laughed happily then",
            "The boy ran fast away",
            "The president spoke firmly today",
            "The chef worked quickly then",
        ],
        "target_words": [
            "wisely", "carefully", "beautifully", "bravely",
            "quickly", "loudly", "softly", "happily",
            "carefully", "clearly", "slowly", "quietly",
            "happily", "fast", "firmly", "quickly",
        ],
    },
}

# 功能词列表
FUNCTION_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'out', 'off', 'over',
    'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 'just', 'because', 'but', 'and',
    'or', 'if', 'while', 'about', 'up', 'down', 'that', 'this', 'these',
    'those', 'it', 'its', 'he', 'she', 'they', 'them', 'we', 'us', 'me',
    'my', 'your', 'his', 'her', 'their', 'our', 'what', 'which', 'who',
    'whom', 'whose', ',', '.', '!', '?', ';', ':', "'", '"', '-',
    "'s", "'t", "'re", "'ve", "'ll", "'d", "'m",
}

ROLE_NAMES = ["nsubj", "dobj", "amod", "advmod"]


def find_token_index(tokens, word):
    word_lower = word.lower()
    for i, tok in enumerate(tokens):
        if word_lower in tok.lower() or tok.lower().startswith(word_lower[:3]):
            return i
    return None


def is_content_word(token_str):
    t = token_str.strip().lower().lstrip('▁').lstrip(' ')
    if not t:
        return False
    if t in FUNCTION_WORDS:
        return False
    if len(t) <= 1:
        return False
    return True


def get_layer_hidden(model, tokenizer, device, sentence, layer_idx):
    """获取指定层的hidden states"""
    layers = get_layers(model)
    target_layer = layers[layer_idx]
    
    toks = tokenizer(sentence, return_tensors="pt").to(device)
    
    captured = {}
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured['h'] = output[0].detach().float().clone()
        else:
            captured['h'] = output.detach().float().clone()
    
    h_handle = target_layer.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        output = model(**toks)
    
    h_handle.remove()
    
    if 'h' not in captured:
        return None, toks
    
    return captured['h'], toks


def get_last_layer_hidden(model, tokenizer, device, sentence):
    """获取最后层的hidden states + logits"""
    layers = get_layers(model)
    last_layer = layers[-1]
    
    toks = tokenizer(sentence, return_tensors="pt").to(device)
    
    captured = {}
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured['h'] = output[0].detach().float().clone()
        else:
            captured['h'] = output.detach().float().clone()
    
    h_handle = last_layer.register_forward_hook(hook_fn)
    
    with torch.no_grad():
        output = model(**toks)
        base_logits = output.logits.detach().float()
    
    h_handle.remove()
    
    if 'h' not in captured:
        return None, None, toks
    
    return captured['h'], base_logits, toks


def collect_hidden_states(model, tokenizer, device, layer_idx, data_dict, role_names):
    """收集指定层所有hidden states"""
    all_hidden = []
    all_labels = []
    
    for role_idx, role in enumerate(role_names):
        data = data_dict[role]
        for sent, target in zip(data["sentences"], data["target_words"]):
            h, toks = get_layer_hidden(model, tokenizer, device, sent, layer_idx)
            if h is None:
                continue
            
            input_ids = toks.input_ids
            tokens = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]
            dep_idx = find_token_index(tokens, target)
            if dep_idx is None:
                continue
            
            h_vec = h[0, dep_idx, :].float().cpu().numpy()
            all_hidden.append(h_vec)
            all_labels.append(role_idx)
    
    return np.array(all_hidden), np.array(all_labels)


def train_probe_and_get_directions(X, y, role_names):
    """训练探针并获取方向(在原始空间中)"""
    if len(set(y)) < 2:
        return None
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    probe = LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0)
    
    cv_folds = min(5, min(np.bincount(y)))
    if cv_folds >= 2:
        cv_scores = cross_val_score(probe, X_scaled, y, cv=cv_folds, scoring='accuracy')
        cv_acc = float(cv_scores.mean())
    else:
        cv_acc = -1.0
    
    probe.fit(X_scaled, y)
    train_acc = float(probe.score(X_scaled, y))
    
    # 提取探针方向(转换回原始空间)
    probe_weights = probe.coef_
    scale_factors = scaler.scale_
    probe_weights_orig = probe_weights / scale_factors[np.newaxis, :]
    
    probe_directions = {}
    for i, role in enumerate(role_names):
        w = probe_weights_orig[i]
        w_norm = np.linalg.norm(w)
        if w_norm > 1e-10:
            probe_directions[role] = w / w_norm
    
    return {
        'probe': probe,
        'scaler': scaler,
        'probe_directions': probe_directions,
        'probe_weights_orig': probe_weights_orig,
        'train_acc': train_acc,
        'cv_acc': cv_acc,
        'n_samples': len(X),
    }


def analyze_tetrahedron(directions, role_names):
    """分析4个方向是否构成正四面体"""
    n = len(role_names)
    dirs = [directions[r] for r in role_names]
    
    # 余弦矩阵
    cos_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cos_matrix[i, j] = np.dot(dirs[i], dirs[j])
    
    # 非对角元素
    off_diag = []
    for i in range(n):
        for j in range(i+1, n):
            off_diag.append(cos_matrix[i, j])
    
    mean_off_diag = np.mean(off_diag)
    std_off_diag = np.std(off_diag)
    
    # 与正四面体的误差
    expected = -1.0 / 3.0
    error = abs(mean_off_diag - expected)
    
    is_tetrahedron = error < 0.05 and std_off_diag < 0.2
    
    # PCA维度分析
    dir_matrix = np.array(dirs)
    pca = PCA(n_components=min(n, 4))
    pca.fit(dir_matrix)
    
    # 到质心的距离
    centroid = np.mean(dir_matrix, axis=0)
    distances = [np.linalg.norm(d - centroid) for d in dirs]
    
    return {
        'cos_matrix': cos_matrix,
        'mean_off_diag': float(mean_off_diag),
        'std_off_diag': float(std_off_diag),
        'expected_tetrahedron': float(expected),
        'error_from_tetrahedron': float(error),
        'is_tetrahedron': is_tetrahedron,
        'pca_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'centroid_distances': [float(d) for d in distances],
        'centroid_dist_cv': float(np.std(distances) / max(np.mean(distances), 1e-10)),
    }


# ================================================================
# Exp1 (8D): 随机标签对照实验
# ================================================================

def exp1_random_label_control(model, tokenizer, device, model_info):
    """
    ★★★★★ 随机标签对照实验
    
    核心问题: 正四面体是探针的artifact, 还是hidden state的真实结构?
    
    方法:
    1. 收集真实hidden states
    2. 随机打乱标签(多次)
    3. 训练探针, 分析方向几何
    4. 对比: 真实标签 vs 随机标签
    
    预测: 如果正四面体是真实结构, 随机标签不应产生正四面体
    """
    print("\n" + "="*70)
    print("Exp1 (8D): 随机标签对照实验 — 正四面体因果验证")
    print("="*70)
    
    # 1. 收集真实hidden states
    print("\n[1] Collecting hidden states from last layer...")
    X_real, y_real = collect_hidden_states(
        model, tokenizer, device, model_info.n_layers - 1, EXTENDED_DATA, ROLE_NAMES
    )
    print(f"  Collected {len(X_real)} samples, {X_real.shape[1]} dims")
    print(f"  Label distribution: {np.bincount(y_real)}")
    
    # 2. 真实标签基线
    print("\n[2] Training probe with REAL labels (baseline)...")
    real_result = train_probe_and_get_directions(X_real, y_real, ROLE_NAMES)
    if real_result is None:
        print("  ERROR: Failed to train real probe")
        return None
    
    real_geo = analyze_tetrahedron(real_result['probe_directions'], ROLE_NAMES)
    print(f"  Real: CV acc={real_result['cv_acc']:.4f}, train_acc={real_result['train_acc']:.4f}")
    print(f"  Real: mean_off_diag={real_geo['mean_off_diag']:.4f}, "
          f"error={real_geo['error_from_tetrahedron']:.4f}, "
          f"is_tetrahedron={real_geo['is_tetrahedron']}")
    
    # 3. 随机标签实验 (10次)
    print("\n[3] Training probes with RANDOM labels (10 trials)...")
    n_trials = 10
    random_results = []
    
    for trial in range(n_trials):
        # 随机打乱标签(保持类别比例)
        y_shuffled = y_real.copy()
        np.random.shuffle(y_shuffled)
        
        # 确保每个类至少有2个样本
        class_counts = np.bincount(y_shuffled)
        if min(class_counts) < 2:
            # 重新分配
            perm = np.random.permutation(len(y_shuffled))
            y_shuffled = y_real[perm]  # 置换真实标签
        
        result = train_probe_and_get_directions(X_real, y_shuffled, ROLE_NAMES)
        if result is None:
            continue
        
        geo = analyze_tetrahedron(result['probe_directions'], ROLE_NAMES)
        random_results.append({
            'trial': trial,
            'cv_acc': result['cv_acc'],
            'train_acc': result['train_acc'],
            'mean_off_diag': geo['mean_off_diag'],
            'error_from_tetrahedron': geo['error_from_tetrahedron'],
            'is_tetrahedron': geo['is_tetrahedron'],
            'std_off_diag': geo['std_off_diag'],
        })
        print(f"  Trial {trial}: CV={result['cv_acc']:.4f}, "
              f"mean_off_diag={geo['mean_off_diag']:.4f}, "
              f"error={geo['error_from_tetrahedron']:.4f}, "
              f"tetra={geo['is_tetrahedron']}")
    
    # 4. 统计分析
    print("\n[4] Statistical Analysis...")
    if random_results:
        random_errors = [r['error_from_tetrahedron'] for r in random_results]
        random_means = [r['mean_off_diag'] for r in random_results]
        random_cv = [r['cv_acc'] for r in random_results]
        random_tetra_count = sum(1 for r in random_results if r['is_tetrahedron'])
        
        print(f"\n  === Real vs Random Comparison ===")
        print(f"  Real:  mean_off_diag={real_geo['mean_off_diag']:.4f}, "
              f"error={real_geo['error_from_tetrahedron']:.4f}")
        print(f"  Random: mean_off_diag={np.mean(random_means):.4f}±{np.std(random_means):.4f}, "
              f"error={np.mean(random_errors):.4f}±{np.std(random_errors):.4f}")
        print(f"  Random tetrahedron count: {random_tetra_count}/{len(random_results)}")
        print(f"  Real CV accuracy: {real_result['cv_acc']:.4f}")
        print(f"  Random CV accuracy: {np.mean(random_cv):.4f}±{np.std(random_cv):.4f}")
        
        # 关键判断
        real_error = real_geo['error_from_tetrahedron']
        random_error_mean = np.mean(random_errors)
        z_score = (real_error - random_error_mean) / max(np.std(random_errors), 1e-10)
        
        print(f"\n  Z-score (real vs random error): {z_score:.2f}")
        if z_score < -2.0:
            print(f"  ★★★ 正四面体是真实结构! (随机标签误差远大于真实标签)")
        elif abs(z_score) < 1.0:
            print(f"  ⚠ 正四面体可能是artifact! (随机标签与真实标签无显著差异)")
        else:
            print(f"  ? 结果不确定, 需要更多实验")
    
    return {
        'real_geo': {k: v for k, v in real_geo.items() if k != 'cos_matrix'},
        'real_cv_acc': float(real_result['cv_acc']),
        'random_results': random_results,
        'random_mean_error': float(np.mean([r['error_from_tetrahedron'] for r in random_results])) if random_results else None,
        'real_error': float(real_geo['error_from_tetrahedron']),
    }


# ================================================================
# Exp2 (8D-ext): 随机向量对照实验
# ================================================================

def exp2_random_vector_control(model, tokenizer, device, model_info):
    """
    ★★★★ 随机向量对照实验
    
    核心问题: LogisticRegression本身是否偏好正四面体?
    
    方法:
    1. 生成随机高斯向量(与真实hidden states同维度)
    2. 用真实标签训练探针
    3. 分析方向几何
    
    预测: 如果LR偏好正四面体, 随机向量也会产生正四面体
    """
    print("\n" + "="*70)
    print("Exp2 (8D-ext): 随机向量对照实验 — LR偏差检验")
    print("="*70)
    
    # 1. 获取真实数据形状
    X_real, y_real = collect_hidden_states(
        model, tokenizer, device, model_info.n_layers - 1, EXTENDED_DATA, ROLE_NAMES
    )
    d_model = X_real.shape[1]
    n_samples = len(X_real)
    print(f"  Real data: {n_samples} samples, {d_model} dims")
    
    # 2. 随机向量实验 (10次)
    print("\n[2] Training probes with RANDOM vectors (10 trials)...")
    n_trials = 10
    random_results = []
    
    for trial in range(n_trials):
        # 生成随机高斯向量
        X_random = np.random.randn(n_samples, d_model)
        
        result = train_probe_and_get_directions(X_random, y_real, ROLE_NAMES)
        if result is None:
            continue
        
        geo = analyze_tetrahedron(result['probe_directions'], ROLE_NAMES)
        random_results.append({
            'trial': trial,
            'cv_acc': result['cv_acc'],
            'train_acc': result['train_acc'],
            'mean_off_diag': geo['mean_off_diag'],
            'error_from_tetrahedron': geo['error_from_tetrahedron'],
            'is_tetrahedron': geo['is_tetrahedron'],
            'std_off_diag': geo['std_off_diag'],
        })
        print(f"  Trial {trial}: CV={result['cv_acc']:.4f}, "
              f"mean_off_diag={geo['mean_off_diag']:.4f}, "
              f"error={geo['error_from_tetrahedron']:.4f}, "
              f"tetra={geo['is_tetrahedron']}")
    
    # 3. 对比真实
    real_result = train_probe_and_get_directions(X_real, y_real, ROLE_NAMES)
    real_geo = analyze_tetrahedron(real_result['probe_directions'], ROLE_NAMES)
    
    print(f"\n  === Real vs Random Vectors ===")
    print(f"  Real:  error={real_geo['error_from_tetrahedron']:.4f}, "
          f"mean_off_diag={real_geo['mean_off_diag']:.4f}")
    if random_results:
        print(f"  Random: error={np.mean([r['error_from_tetrahedron'] for r in random_results]):.4f}±"
              f"{np.std([r['error_from_tetrahedron'] for r in random_results]):.4f}")
        random_tetra_count = sum(1 for r in random_results if r['is_tetrahedron'])
        print(f"  Random tetrahedron count: {random_tetra_count}/{len(random_results)}")
    
    return {
        'real_error': float(real_geo['error_from_tetrahedron']),
        'random_results': random_results,
    }


# ================================================================
# Exp3 (8B): 语义-语法解耦操控
# ================================================================

def exp3_semantic_grammar_decoupling(model, tokenizer, device, model_info):
    """
    ★★★★★ 语义-语法解耦操控
    
    核心问题: 探针方向中的"纯语法"成分是什么?
    
    方法:
    1. 获取W_U矩阵 [vocab_size, d_model]
    2. 对每个探针方向w_i, 分解:
       - w_i_parallel = proj_{W_U}(w_i)  — W_U行空间中的分量
       - w_i_perp = w_i - w_i_parallel    — W_U行空间正交补中的分量
    3. 测试: 
       - 添加w_i_perp是否只改变语法分类?
       - 添加w_i_parallel是否主要改变token选择?
    4. 对比效果:
       - w_i (原方向): 改变分类+改变token
       - w_i_perp (纯语法): 只改变分类, 不改变token
       - w_i_parallel (语义): 改变token, 可能改变分类
    """
    print("\n" + "="*70)
    print("Exp3 (8B): 语义-语法解耦操控")
    print("="*70)
    
    # 1. 训练探针
    print("\n[1] Training probe on last layer...")
    X_real, y_real = collect_hidden_states(
        model, tokenizer, device, model_info.n_layers - 1, EXTENDED_DATA, ROLE_NAMES
    )
    probe_result = train_probe_and_get_directions(X_real, y_real, ROLE_NAMES)
    if probe_result is None:
        print("  ERROR: Failed to train probe")
        return None
    
    probe = probe_result['probe']
    scaler = probe_result['scaler']
    probe_directions = probe_result['probe_directions']
    print(f"  Probe trained: CV={probe_result['cv_acc']:.4f}")
    
    # 2. 获取W_U并计算行空间基
    print("\n[2] Computing W_U row space basis...")
    W_U = get_W_U(model)  # [vocab_size, d_model]
    print(f"  W_U shape: {W_U.shape}")
    
    # SVD of W_U^T to get row space basis
    W_U_T = W_U.T.astype(np.float32)  # [d_model, vocab_size]
    k = min(200, min(W_U_T.shape[0], W_U_T.shape[1]) - 2)
    print(f"  Computing SVD of W_U^T with k={k}...")
    U_wut, s_wut, _ = svds(W_U_T, k=k)
    U_wut = np.asarray(U_wut, dtype=np.float64)  # [d_model, k]
    print(f"  SVD done: U shape={U_wut.shape}")
    print(f"  Top 5 singular values: {s_wut[:5]}")
    print(f"  Explained variance by top {k} components: "
          f"{float(np.sum(s_wut**2) / np.sum(np.linalg.svd(W_U_T, compute_uv=False)**2)):.4f}")
    
    # 3. 分解每个探针方向
    print("\n[3] Decomposing probe directions...")
    decomposition = {}
    for role in ROLE_NAMES:
        w = probe_directions[role]  # [d_model], unit norm
        
        # 投影到W_U行空间
        proj_coeffs = U_wut.T @ w  # [k]
        w_parallel = U_wut @ proj_coeffs  # [d_model]
        w_perp = w - w_parallel
        
        # 各分量能量
        total_energy = np.dot(w, w)
        parallel_energy = np.dot(w_parallel, w_parallel)
        perp_energy = np.dot(w_perp, w_perp)
        
        # 归一化
        w_parallel_norm = np.linalg.norm(w_parallel)
        w_perp_norm = np.linalg.norm(w_perp)
        
        if w_parallel_norm > 1e-10:
            w_parallel_hat = w_parallel / w_parallel_norm
        else:
            w_parallel_hat = w_parallel
        
        if w_perp_norm > 1e-10:
            w_perp_hat = w_perp / w_perp_norm
        else:
            w_perp_hat = w_perp
        
        decomposition[role] = {
            'w': w,
            'w_parallel': w_parallel,
            'w_perp': w_perp,
            'w_parallel_hat': w_parallel_hat,
            'w_perp_hat': w_perp_hat,
            'parallel_ratio': float(parallel_energy / max(total_energy, 1e-20)),
            'perp_ratio': float(perp_energy / max(total_energy, 1e-20)),
            'w_parallel_norm': float(w_parallel_norm),
            'w_perp_norm': float(w_perp_norm),
        }
        
        print(f"  {role}: parallel={parallel_energy/total_energy:.4f}, "
              f"perp={perp_energy/total_energy:.4f}, "
              f"||w_par||={w_parallel_norm:.4f}, ||w_perp||={w_perp_norm:.4f}")
    
    # 4. 测试操控效果
    print("\n[4] Testing manipulation effects...")
    
    manipulation_tests = [
        ("The cat sat on the mat", "cat", "nsubj", "dobj"),
        ("She chased the dog away", "dog", "dobj", "nsubj"),
        ("The beautiful bird sang softly", "beautiful", "amod", "advmod"),
        ("The cat ran quickly home", "quickly", "advmod", "amod"),
    ]
    
    alphas = [0.3, 0.5, 1.0, 2.0]
    
    results_by_direction = {
        'original': {'flip': 0, 'total': 0, 'content_words': 0, 'kl_sum': 0},
        'perp': {'flip': 0, 'total': 0, 'content_words': 0, 'kl_sum': 0},
        'parallel': {'flip': 0, 'total': 0, 'content_words': 0, 'kl_sum': 0},
    }
    
    for sent, target, src_role, tgt_role in manipulation_tests:
        print(f"\n  [{src_role}→{tgt_role}] {sent} / '{target}'")
        
        h, base_logits, toks = get_last_layer_hidden(model, tokenizer, device, sent)
        if h is None:
            continue
        
        input_ids = toks.input_ids
        tokens_list = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]
        dep_idx = find_token_index(tokens_list, target)
        if dep_idx is None:
            continue
        
        h_vec = h[0, dep_idx, :].float().cpu().numpy()
        base_probs = torch.softmax(base_logits[0, dep_idx].float(), dim=-1).cpu().numpy()
        base_top_idx = np.argmax(base_probs)
        base_top_token = safe_decode(tokenizer, int(base_top_idx))
        
        # 原始探针预测
        h_scaled = scaler.transform(h_vec.reshape(1, -1))
        orig_pred = int(probe.predict(h_scaled)[0])
        orig_role = ROLE_NAMES[orig_pred]
        
        # 三种方向
        dec = decomposition[tgt_role]
        directions = {
            'original': dec['w'],
            'perp': dec['w_perp_hat'] if dec['w_perp_norm'] > 0.01 else None,
            'parallel': dec['w_parallel_hat'] if dec['w_parallel_norm'] > 0.01 else None,
        }
        
        for dir_name, direction in directions.items():
            if direction is None:
                print(f"    {dir_name}: SKIPPED (norm too small)")
                continue
            
            for alpha in alphas:
                # 操控
                h_new = h_vec + alpha * direction
                h_new_scaled = scaler.transform(h_new.reshape(1, -1))
                new_pred = int(probe.predict(h_new_scaled)[0])
                new_role = ROLE_NAMES[new_pred]
                flipped = (new_pred != orig_pred)
                
                # 计算logits变化
                h_tensor = torch.tensor(h_new, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                
                # 用final norm + lm_head
                if hasattr(model, 'model') and hasattr(model.model, 'norm'):
                    normed = model.model.norm(h_tensor.to(model.model.norm.weight.device).to(model.model.norm.weight.dtype))
                else:
                    normed = h_tensor
                
                if hasattr(model, 'lm_head'):
                    new_logits = model.lm_head(normed.to(model.lm_head.weight.dtype))
                else:
                    new_logits = normed
                
                new_logits = new_logits.detach().float().cpu().numpy()[0, 0]
                new_probs = np.exp(new_logits - np.max(new_logits))
                new_probs = new_probs / new_probs.sum()
                
                new_top_idx = np.argmax(new_probs)
                new_top_token = safe_decode(tokenizer, int(new_top_idx))
                
                # KL散度
                kl = np.sum(base_probs * (np.log(base_probs + 1e-10) - np.log(new_probs + 1e-10)))
                
                is_content = is_content_word(new_top_token)
                
                if flipped:
                    results_by_direction[dir_name]['flip'] += 1
                    results_by_direction[dir_name]['total'] += 1
                    results_by_direction[dir_name]['kl_sum'] += kl
                    if is_content:
                        results_by_direction[dir_name]['content_words'] += 1
                else:
                    results_by_direction[dir_name]['total'] += 1
                
                if alpha in [0.5, 1.0] and flipped:
                    print(f"    {dir_name} α={alpha}: {orig_role}→{new_role}, "
                          f"top: {base_top_token}→{new_top_token} "
                          f"({'content' if is_content else 'function'}), "
                          f"KL={kl:.3f}")
    
    # 5. 汇总
    print("\n[5] Summary of direction decomposition effects:")
    for dir_name, stats in results_by_direction.items():
        flip_rate = stats['flip'] / max(stats['total'], 1)
        content_rate = stats['content_words'] / max(stats['flip'], 1)
        mean_kl = stats['kl_sum'] / max(stats['flip'], 1)
        print(f"  {dir_name}: flip_rate={flip_rate:.1%}, "
              f"content_word_rate={content_rate:.1%}, "
              f"mean_KL={mean_kl:.3f}")
    
    # 6. 正四面体检验: perp方向是否也构成正四面体?
    print("\n[6] Tetrahedron check for decomposed directions...")
    
    # 检查w_perp_hat方向
    perp_dirs = {}
    for role in ROLE_NAMES:
        if decomposition[role]['w_perp_norm'] > 0.01:
            perp_dirs[role] = decomposition[role]['w_perp_hat']
    
    if len(perp_dirs) == 4:
        perp_geo = analyze_tetrahedron(perp_dirs, ROLE_NAMES)
        print(f"  w_perp: mean_off_diag={perp_geo['mean_off_diag']:.4f}, "
              f"error={perp_geo['error_from_tetrahedron']:.4f}, "
              f"tetra={perp_geo['is_tetrahedron']}")
    else:
        perp_geo = None
        print(f"  w_perp: only {len(perp_dirs)} valid directions, skipping")
    
    # 检查w_parallel_hat方向
    par_dirs = {}
    for role in ROLE_NAMES:
        if decomposition[role]['w_parallel_norm'] > 0.01:
            par_dirs[role] = decomposition[role]['w_parallel_hat']
    
    if len(par_dirs) == 4:
        par_geo = analyze_tetrahedron(par_dirs, ROLE_NAMES)
        print(f"  w_parallel: mean_off_diag={par_geo['mean_off_diag']:.4f}, "
              f"error={par_geo['error_from_tetrahedron']:.4f}, "
              f"tetra={par_geo['is_tetrahedron']}")
    else:
        par_geo = None
        print(f"  w_parallel: only {len(par_dirs)} valid directions, skipping")
    
    return {
        'decomposition': {role: {
            'parallel_ratio': dec['parallel_ratio'],
            'perp_ratio': dec['perp_ratio'],
            'w_parallel_norm': dec['w_parallel_norm'],
            'w_perp_norm': dec['w_perp_norm'],
        } for role, dec in decomposition.items()},
        'results_by_direction': {k: {
            'flip_rate': v['flip'] / max(v['total'], 1),
            'content_word_rate': v['content_words'] / max(v['flip'], 1),
            'mean_kl': v['kl_sum'] / max(v['flip'], 1),
        } for k, v in results_by_direction.items()},
        'perp_tetrahedron': {k: v for k, v in perp_geo.items() if k != 'cos_matrix'} if perp_geo else None,
        'parallel_tetrahedron': {k: v for k, v in par_geo.items() if k != 'cos_matrix'} if par_geo else None,
    }


# ================================================================
# Exp4 (8A): 正四面体对称性验证
# ================================================================

def exp4_symmetry_verification(model, tokenizer, device, model_info):
    """
    ★★★★ 正四面体对称性验证
    
    核心问题: 4个方向的对称性有多强?
    
    方法:
    1. 测量各角色的分类margin
    2. 测量各角色的类内方差
    3. 测量各角色对的类间距离
    4. 置换等变性: 交换标签后探针是否同样准确?
    """
    print("\n" + "="*70)
    print("Exp4 (8A): 正四面体对称性验证")
    print("="*70)
    
    # 1. 收集hidden states
    print("\n[1] Collecting hidden states...")
    X_real, y_real = collect_hidden_states(
        model, tokenizer, device, model_info.n_layers - 1, EXTENDED_DATA, ROLE_NAMES
    )
    print(f"  {len(X_real)} samples")
    
    # 2. 训练探针
    print("\n[2] Training probe...")
    probe_result = train_probe_and_get_directions(X_real, y_real, ROLE_NAMES)
    if probe_result is None:
        print("  ERROR: Failed to train probe")
        return None
    
    probe = probe_result['probe']
    scaler = probe_result['scaler']
    print(f"  CV accuracy: {probe_result['cv_acc']:.4f}")
    
    # 3. 各角色的类内统计
    print("\n[3] Per-class statistics...")
    X_scaled = scaler.transform(X_real)
    
    class_stats = {}
    for role_idx, role in enumerate(ROLE_NAMES):
        mask = y_real == role_idx
        X_class = X_scaled[mask]
        
        # 类中心
        center = X_class.mean(axis=0)
        
        # 类内方差
        var = np.mean(np.sum((X_class - center)**2, axis=1))
        
        # 分类margin (到决策边界的距离)
        if hasattr(probe, 'decision_function'):
            margins = probe.decision_function(X_class)
            # margin for the correct class
            correct_margin = np.mean([m[role_idx] for m in margins])
        else:
            correct_margin = -1
        
        class_stats[role] = {
            'n_samples': int(mask.sum()),
            'intra_var': float(var),
            'mean_margin': float(correct_margin),
        }
        print(f"  {role}: n={mask.sum()}, intra_var={var:.4f}, margin={correct_margin:.4f}")
    
    # 4. 类间距离矩阵
    print("\n[4] Inter-class distance matrix...")
    centers = {}
    for role_idx, role in enumerate(ROLE_NAMES):
        mask = y_real == role_idx
        centers[role] = X_scaled[mask].mean(axis=0)
    
    dist_matrix = np.zeros((4, 4))
    for i, r1 in enumerate(ROLE_NAMES):
        for j, r2 in enumerate(ROLE_NAMES):
            dist_matrix[i, j] = np.linalg.norm(centers[r1] - centers[r2])
    
    print("  Distance matrix:")
    print("         " + "  ".join(f"{r:>8s}" for r in ROLE_NAMES))
    for i, r1 in enumerate(ROLE_NAMES):
        row = f"  {r1:>6s}"
        for j in range(4):
            row += f"  {dist_matrix[i,j]:8.4f}"
        print(row)
    
    # 类间距离的变异系数 (衡量对称性)
    off_diag_dists = []
    for i in range(4):
        for j in range(i+1, 4):
            off_diag_dists.append(dist_matrix[i, j])
    
    dist_cv = np.std(off_diag_dists) / max(np.mean(off_diag_dists), 1e-10)
    print(f"  Distance CV: {dist_cv:.4f} (0 = perfect symmetry)")
    
    # 5. 置换等变性测试
    print("\n[5] Permutation equivariance test...")
    
    # 选择几个代表性置换
    permutations = {
        'identity': [0, 1, 2, 3],
        'swap_01': [1, 0, 2, 3],  # swap nsubj <-> dobj
        'swap_23': [0, 1, 3, 2],  # swap amod <-> advmod
        'swap_02': [2, 1, 0, 3],  # swap nsubj <-> amod
        'swap_13': [0, 3, 2, 1],  # swap dobj <-> advmod
        'rotate': [1, 2, 3, 0],   # cyclic: nsubj->dobj->advmod->amod->nsubj
    }
    
    perm_results = {}
    for perm_name, perm in permutations.items():
        y_perm = np.array([perm[y] for y in y_real])
        
        # 确保每个类有足够样本
        counts = np.bincount(y_perm)
        if min(counts) < 2:
            print(f"  {perm_name}: SKIPPED (insufficient samples)")
            continue
        
        result = train_probe_and_get_directions(X_real, y_perm, ROLE_NAMES)
        if result is None:
            continue
        
        # 分析几何
        geo = analyze_tetrahedron(result['probe_directions'], ROLE_NAMES)
        
        perm_results[perm_name] = {
            'cv_acc': result['cv_acc'],
            'mean_off_diag': geo['mean_off_diag'],
            'error_from_tetrahedron': geo['error_from_tetrahedron'],
            'is_tetrahedron': geo['is_tetrahedron'],
        }
        print(f"  {perm_name}: CV={result['cv_acc']:.4f}, "
              f"error={geo['error_from_tetrahedron']:.4f}, "
              f"tetra={geo['is_tetrahedron']}")
    
    # 6. 综合对称性评分
    print("\n[6] Overall symmetry score...")
    
    # 各维度对称性
    margin_var = np.var([s['mean_margin'] for s in class_stats.values()])
    var_var = np.var([s['intra_var'] for s in class_stats.values()])
    
    symmetry_scores = {
        'distance_cv': float(dist_cv),
        'margin_variance': float(margin_var),
        'intra_var_variance': float(var_var),
        'perm_tetrahedron_rate': sum(1 for r in perm_results.values() if r['is_tetrahedron']) / max(len(perm_results), 1),
        'perm_cv_mean': float(np.mean([r['cv_acc'] for r in perm_results.values()])) if perm_results else 0,
    }
    
    print(f"  Distance CV: {symmetry_scores['distance_cv']:.4f}")
    print(f"  Margin variance: {symmetry_scores['margin_variance']:.6f}")
    print(f"  Intra-var variance: {symmetry_scores['intra_var_variance']:.4f}")
    print(f"  Permutation tetrahedron rate: {symmetry_scores['perm_tetrahedron_rate']:.1%}")
    print(f"  Permutation CV mean: {symmetry_scores['perm_cv_mean']:.4f}")
    
    overall_symmetry = 1.0 - min(symmetry_scores['distance_cv'], 1.0)
    print(f"  Overall symmetry: {overall_symmetry:.4f} (1 = perfect)")
    
    return {
        'class_stats': class_stats,
        'distance_cv': float(dist_cv),
        'off_diag_distances': [float(d) for d in off_diag_dists],
        'perm_results': perm_results,
        'symmetry_scores': symmetry_scores,
    }


# ================================================================
# 主函数
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="CCL-K: Tetrahedron Verification")
    parser.add_argument("--model", type=str, required=True, 
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, required=True, 
                       choices=[1, 2, 3, 4],
                       help="1=random_label, 2=random_vector, 3=decoupling, 4=symmetry")
    args = parser.parse_args()
    
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    print(f"Model: {model_info.name}, layers={model_info.n_layers}, d_model={model_info.d_model}")
    
    exp_funcs = {
        1: exp1_random_label_control,
        2: exp2_random_vector_control,
        3: exp3_semantic_grammar_decoupling,
        4: exp4_symmetry_verification,
    }
    
    result = exp_funcs[args.exp](model, tokenizer, device, model_info)
    
    # 保存结果
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'glm5_temp')
    os.makedirs(results_dir, exist_ok=True)
    
    exp_names = {1: "random_label", 2: "random_vector", 3: "decoupling", 4: "symmetry"}
    result_path = os.path.join(results_dir, 
                              f"cclk_exp{args.exp}_{exp_names[args.exp]}_{model_info.name}_results.json")
    
    # 转换numpy为可序列化
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
    
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(convert(result), f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {result_path}")
    
    release_model(model)


if __name__ == "__main__":
    main()
