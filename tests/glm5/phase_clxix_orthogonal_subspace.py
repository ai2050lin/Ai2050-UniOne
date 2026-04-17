"""
Phase CLXIX: W_U正交子空间分解 — 验证正交性→全局唯一性
P728: W_U的SVD子空间与语言维度的对应
  - 前50个主成分 → 风格/逻辑/语法的映射
  - 子空间正交性检验: 风格子空间 ⊥ 逻辑子空间?
  - 如果正交 → 正交性→独立性 → 全局唯一性有数学基础!

核心思路:
  如果W_U的行可以被分解为正交子空间, 每个子空间对应一个语言维度,
  那么"所有神经元参与但输出唯一"就有了解释:
    W_U = [风格子空间 | 逻辑子空间 | 语法子空间 | 残差]
    正交 → 修改一个维度不影响其他 → 多维同时控制!
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import json
import argparse
from scipy import stats
from sklearn.decomposition import PCA
from collections import defaultdict

from model_utils import load_model, get_model_info, get_W_U


def to_numpy(tensor_or_array):
    """统一转换为numpy float32数组"""
    if isinstance(tensor_or_array, np.ndarray):
        return tensor_or_array.astype(np.float32)
    return tensor_or_array.detach().cpu().numpy().astype(np.float32)

# ============================================================
# 词分类：按语义/语法/风格/逻辑维度标注
# ============================================================

# 语法角色分类
SYNTAX_CATS = {
    'noun': ['cat', 'dog', 'house', 'car', 'tree', 'book', 'water', 'food',
             'hand', 'head', 'eye', 'door', 'table', 'chair', 'window',
             'apple', 'sun', 'stone', 'hair', 'fire', 'mountain', 'river',
             'king', 'queen', 'man', 'woman', 'boy', 'girl', 'child', 'father',
             'mother', 'brother', 'sister', 'friend', 'teacher', 'doctor'],
    'verb': ['run', 'walk', 'eat', 'drink', 'see', 'hear', 'think', 'know',
             'make', 'take', 'give', 'come', 'go', 'sit', 'stand', 'write',
             'read', 'speak', 'listen', 'feel', 'love', 'hate', 'want', 'need',
             'can', 'will', 'should', 'must', 'may', 'might', 'could', 'would'],
    'adj': ['big', 'small', 'good', 'bad', 'hot', 'cold', 'new', 'old',
            'fast', 'slow', 'high', 'low', 'long', 'short', 'hard', 'soft',
            'red', 'blue', 'green', 'white', 'black', 'dark', 'bright', 'clean'],
    'prep': ['in', 'on', 'at', 'to', 'from', 'with', 'by', 'for',
             'of', 'about', 'into', 'through', 'between', 'under', 'over', 'after'],
    'det': ['the', 'a', 'an', 'this', 'that', 'these', 'those', 'my',
            'your', 'his', 'her', 'its', 'our', 'their', 'some', 'any'],
    'pron': ['i', 'you', 'he', 'she', 'it', 'we', 'they', 'me',
             'him', 'us', 'them', 'who', 'what', 'which', 'where', 'when'],
    'conj': ['and', 'but', 'or', 'nor', 'so', 'yet', 'for', 'because',
             'if', 'when', 'while', 'although', 'since', 'unless', 'until', 'whether'],
    'adv': ['very', 'really', 'quite', 'rather', 'almost', 'never', 'always',
            'often', 'sometimes', 'usually', 'already', 'still', 'also', 'just',
            'only', 'even', 'too', 'again', 'here', 'there', 'now', 'then'],
}

# 语义领域分类(逻辑维度)
SEMANTIC_DOMAINS = {
    'physical': ['water', 'fire', 'stone', 'tree', 'mountain', 'river', 'sun',
                 'rain', 'wind', 'snow', 'earth', 'sky', 'sea', 'land', 'air',
                 'hot', 'cold', 'big', 'small', 'hard', 'soft'],
    'social': ['king', 'queen', 'man', 'woman', 'child', 'friend', 'teacher',
               'doctor', 'mother', 'father', 'brother', 'sister', 'love', 'hate',
               'good', 'bad', 'help', 'fight', 'work', 'play'],
    'abstract': ['think', 'know', 'believe', 'understand', 'remember', 'forget',
                 'idea', 'concept', 'truth', 'false', 'reason', 'logic', 'time',
                 'space', 'cause', 'effect', 'possible', 'necessary'],
    'action': ['run', 'walk', 'eat', 'drink', 'make', 'take', 'give', 'come',
               'go', 'sit', 'stand', 'write', 'read', 'speak', 'build', 'break'],
}

# 风格标记
STYLE_MARKERS = {
    'formal': ['therefore', 'consequently', 'furthermore', 'nevertheless', 'however',
               'regarding', 'pursuant', 'hereby', 'aforementioned', 'notwithstanding'],
    'informal': ['gonna', 'wanna', 'gotta', 'kinda', 'sorta', 'yeah', 'nah',
                 'cool', 'awesome', 'wow', 'hey', 'huh', 'oops', 'yay'],
    'poetic': ['bloom', 'whisper', 'shadow', 'dream', 'eternal', 'twilight',
               'blossom', 'serene', 'melancholy', 'radiant', 'gossamer', 'luminous'],
}


def tokenize_words(words, tokenizer, model_name):
    """将词列表转换为token ID"""
    word_ids = {}
    for w in words:
        # 尝试直接编码
        if hasattr(tokenizer, 'encode'):
            try:
                ids = tokenizer.encode(w, add_special_tokens=False)
                if len(ids) == 1:
                    word_ids[w] = ids[0]
            except:
                pass
        # 也尝试加空格
        for prefix in ['', ' ', '▁']:
            try:
                ids = tokenizer.encode(prefix + w, add_special_tokens=False)
                if len(ids) == 1:
                    word_ids[w] = ids[0]
                    break
            except:
                pass
    return word_ids


def get_word_vectors(word_ids, W_U):
    """从W_U获取词向量"""
    W_U_np = to_numpy(W_U) if not isinstance(W_U, np.ndarray) else W_U.astype(np.float32)
    vectors = {}
    for w, idx in word_ids.items():
        if idx < W_U_np.shape[0]:
            vectors[w] = W_U_np[idx]
    return vectors


def analyze_subspace_alignment(vectors_dict, W_U_np, n_pc=50):
    """
    P728核心: 分析W_U的SVD子空间与语言维度的对应
    
    方法:
    1. 对W_U做PCA, 取前n_pc个主成分
    2. 对每个语言维度(语法/语义/风格), 计算该维度的词向量在各PC上的投影分布
    3. 分析各PC是否偏好特定语言维度
    4. 检验不同语言维度的子空间是否正交
    """
    print("\n=== P728: W_U正交子空间分解 ===")
    
    d = W_U_np.shape[1]
    
    # Step 1: PCA分解
    print(f"  PCA分解W_U (shape={W_U_np.shape}, n_pc={n_pc})...")
    pca = PCA(n_components=min(n_pc, d))
    pca.fit(W_U_np.astype(np.float32))
    components = pca.components_  # [n_pc, d]
    explained_var = pca.explained_variance_ratio_
    
    print(f"  前10个PC的方差解释率: {explained_var[:10].round(4)}")
    print(f"  前{n_pc}个PC总方差解释率: {explained_var.sum():.4f}")
    
    results = {
        'n_pc': n_pc,
        'explained_var_top10': explained_var[:10].tolist(),
        'total_var_explained': float(explained_var.sum()),
    }
    
    # Step 2: 计算各类词在各PC上的投影分布
    print("\n  计算各类词在PC上的投影分布...")
    
    # 合并所有词分类
    all_cats = {}
    all_cats.update({f'syntax_{k}': v for k, v in SYNTAX_CATS.items()})
    all_cats.update({f'semantic_{k}': v for k, v in SEMANTIC_DOMAINS.items()})
    all_cats.update({f'style_{k}': v for k, v in STYLE_MARKERS.items()})
    
    cat_projections = {}  # cat_name -> {pc_idx: [projections]}
    cat_mean_proj = {}    # cat_name -> mean projection per PC
    
    for cat_name, words in all_cats.items():
        vecs = [vectors_dict[w] for w in words if w in vectors_dict]
        if len(vecs) < 2:
            continue
        
        vecs = np.array(vecs)  # [n_words, d]
        # 投影到各PC
        proj = vecs @ components.T  # [n_words, n_pc]
        cat_projections[cat_name] = proj
        cat_mean_proj[cat_name] = proj.mean(axis=0)
    
    results['n_categories'] = len(cat_mean_proj)
    results['categories'] = list(cat_mean_proj.keys())
    
    # Step 3: 分析各PC的语言维度偏好
    print("\n  分析各PC的语言维度偏好...")
    
    pc_preferences = []  # 每个PC最偏好的类别
    for pc_idx in range(min(n_pc, len(components))):
        # 各类别在该PC上的平均投影(绝对值)
        cat_scores = {}
        for cat_name, proj in cat_projections.items():
            cat_scores[cat_name] = np.abs(proj[:, pc_idx]).mean()
        
        # 排序
        sorted_cats = sorted(cat_scores.items(), key=lambda x: -x[1])
        top_cat = sorted_cats[0][0]
        top_score = sorted_cats[0][1]
        
        # 区分偏好类型
        if top_cat.startswith('syntax_'):
            pref_type = 'syntax'
        elif top_cat.startswith('semantic_'):
            pref_type = 'semantic'
        elif top_cat.startswith('style_'):
            pref_type = 'style'
        else:
            pref_type = 'unknown'
        
        pc_preferences.append({
            'pc_idx': pc_idx,
            'var_explained': float(explained_var[pc_idx]),
            'top_category': top_cat,
            'top_score': float(top_score),
            'preference_type': pref_type,
            'top3_categories': [(c, float(s)) for c, s in sorted_cats[:3]],
        })
    
    results['pc_preferences_top20'] = pc_preferences[:20]
    
    # 统计各PC的偏好类型分布
    pref_counts = defaultdict(int)
    for pp in pc_preferences:
        pref_counts[pp['preference_type']] += 1
    
    print(f"\n  PC偏好类型分布:")
    for ptype, count in sorted(pref_counts.items()):
        print(f"    {ptype}: {count}/{n_pc} ({count/n_pc*100:.1f}%)")
    
    results['pc_preference_distribution'] = dict(pref_counts)
    
    # Step 4: 子空间正交性检验
    print("\n  子空间正交性检验...")
    
    # 构建三类子空间: syntax / semantic / style
    subspace_words = {
        'syntax': [],
        'semantic': [],
        'style': [],
    }
    
    for cat_name, words in SYNTAX_CATS.items():
        subspace_words['syntax'].extend(words)
    for cat_name, words in SEMANTIC_DOMAINS.items():
        subspace_words['semantic'].extend(words)
    for cat_name, words in STYLE_MARKERS.items():
        subspace_words['style'].extend(words)
    
    # 构建子空间的主方向
    subspace_directions = {}
    for dim_name, words in subspace_words.items():
        vecs = [vectors_dict[w] for w in words if w in vectors_dict]
        if len(vecs) < 2:
            continue
        vecs = np.array(vecs)
        # PCA得到该维度的主方向
        dim_pca = PCA(n_components=min(10, vecs.shape[0]-1, vecs.shape[1]))
        dim_pca.fit(vecs)
        subspace_directions[dim_name] = dim_pca.components_  # [n_dim_pc, d]
    
    # 计算子空间间的正交性
    orthogonality_results = {}
    dim_names = list(subspace_directions.keys())
    for i in range(len(dim_names)):
        for j in range(i+1, len(dim_names)):
            dim_i = subspace_directions[dim_names[i]]
            dim_j = subspace_directions[dim_names[j]]
            
            # 子空间间的余弦相似度矩阵
            cos_matrix = np.abs(dim_i @ dim_j.T)  # [n_i, n_j]
            mean_cos = float(cos_matrix.mean())
            max_cos = float(cos_matrix.max())
            
            # 子空间角度(用SVD of cross-projection)
            cross_proj = dim_i @ dim_j.T  # [n_i, n_j]
            svd_vals = np.linalg.svd(cross_proj, compute_uv=False)
            # 子空间角度 = arccos(svd_vals)
            principal_angles = np.arccos(np.clip(svd_vals, 0, 1)) * 180 / np.pi
            min_angle = float(principal_angles.min())
            mean_angle = float(principal_angles.mean())
            
            orth_key = f"{dim_names[i]}_vs_{dim_names[j]}"
            orthogonality_results[orth_key] = {
                'mean_cos': mean_cos,
                'max_cos': max_cos,
                'min_angle_deg': min_angle,
                'mean_angle_deg': mean_angle,
                'principal_angles': principal_angles.tolist(),
            }
            
            print(f"    {orth_key}: mean_cos={mean_cos:.4f}, "
                  f"min_angle={min_angle:.1f}°, mean_angle={mean_angle:.1f}°")
    
    results['subspace_orthogonality'] = orthogonality_results
    
    # Step 5: 正交性→独立性验证(理论推导)
    print("\n  正交性→独立性理论推导:")
    
    # 如果W_U行正交(cos≈0.004), 则:
    # logit(i) = h · W_U[i] 和 logit(j) = h · W_U[j] 几乎独立
    # 因为 Cov(logit_i, logit_j) = h^T · W_U[i] · W_U[j]^T · h ≈ 0
    
    # 实际计算: W_U行间余弦的分布
    n_sample = min(5000, W_U_np.shape[0])
    sample_idx = np.random.choice(W_U_np.shape[0], n_sample, replace=False)
    sample_W = W_U_np[sample_idx]
    
    # 采样计算行间余弦
    n_pairs = 10000
    cos_values = []
    for _ in range(n_pairs):
        i, j = np.random.choice(n_sample, 2, replace=False)
        cos_val = np.dot(sample_W[i], sample_W[j]) / (
            np.linalg.norm(sample_W[i]) * np.linalg.norm(sample_W[j]) + 1e-10)
        cos_values.append(cos_val)
    
    cos_values = np.array(cos_values)
    cos_mean = float(np.abs(cos_values).mean())
    cos_std = float(np.abs(cos_values).std())
    cos_max = float(np.abs(cos_values).max())
    
    # 与随机矩阵对比
    random_W = np.random.randn(n_sample, d).astype(np.float32)
    random_W = random_W / np.linalg.norm(random_W, axis=1, keepdims=True) * \
               np.linalg.norm(sample_W, axis=1, keepdims=True)
    
    random_cos = []
    for _ in range(n_pairs):
        i, j = np.random.choice(n_sample, 2, replace=False)
        cos_val = np.dot(random_W[i], random_W[j]) / (
            np.linalg.norm(random_W[i]) * np.linalg.norm(random_W[j]) + 1e-10)
        random_cos.append(cos_val)
    
    random_cos = np.array(random_cos)
    random_cos_mean = float(np.abs(random_cos).mean())
    
    print(f"    W_U行间|cos|: mean={cos_mean:.4f}, std={cos_std:.4f}, max={cos_max:.4f}")
    print(f"    随机矩阵|cos|: mean={random_cos_mean:.4f}")
    print(f"    正交性提升: {random_cos_mean/cos_mean:.2f}x (W_U比随机更正交)")
    
    # 统计检验: W_U的cos是否显著小于随机
    t_stat, p_val = stats.ttest_ind(np.abs(cos_values), np.abs(random_cos))
    print(f"    t检验: t={t_stat:.2f}, p={p_val:.2e}")
    
    results['orthogonality_analysis'] = {
        'W_U_cos_mean': cos_mean,
        'W_U_cos_std': cos_std,
        'W_U_cos_max': cos_max,
        'random_cos_mean': random_cos_mean,
        'orthogonality_enhancement': float(random_cos_mean / cos_mean) if cos_mean > 0 else 0,
        't_test_t': float(t_stat),
        't_test_p': float(p_val),
        'conclusion': 'W_U比随机更正交→训练强化了正交性' if cos_mean < random_cos_mean else 'W_U不比随机更正交',
    }
    
    # Step 6: 正交性→全局唯一性的定量检验
    print("\n  正交性→全局唯一性检验:")
    
    # 如果W_U行正交, 则修改h使logit(i)增加Δ → logit(j)的变化≈0
    # 量化: Cov(logit_i, logit_j) = h^T W_U[i] W_U[j]^T h
    
    # 选10对词, 计算logit间的相关
    test_words = ['cat', 'dog', 'king', 'queen', 'run', 'walk', 'big', 'small', 'the', 'and']
    test_word_ids = {}
    for w in test_words:
        if w in vectors_dict:
            test_word_ids[w] = list(vectors_dict.keys()).index(w)
    
    # 生成随机h, 计算logit间的相关性
    n_h_samples = 1000
    test_vecs = np.array([vectors_dict[w] for w in test_words if w in vectors_dict])  # [n_words, d]
    
    if len(test_vecs) > 2:
        # 生成随机h
        h_samples = np.random.randn(n_h_samples, d).astype(np.float32)
        
        # 计算logits
        logits = h_samples @ test_vecs.T  # [n_h_samples, n_words]
        
        # 计算logit间的相关
        logit_corr = np.corrcoef(logits.T)  # [n_words, n_words]
        
        # 非对角线元素
        n = logit_corr.shape[0]
        mask = ~np.eye(n, dtype=bool)
        off_diag = logit_corr[mask]
        
        print(f"    logit间相关: mean={np.abs(off_diag).mean():.4f}, "
              f"std={off_diag.std():.4f}, max_abs={np.abs(off_diag).max():.4f}")
        
        # 理论预期: 如果W_U行正交(cos≈0), 则logit相关≈0
        # 实际: logit相关 = W_U[i]·W_U[j]·(h的变化方向)
        # 平均来说, logit相关 ≈ cos(W_U[i], W_U[j])
        
        # 直接计算W_U行间余弦
        wu_cos_matrix = np.abs(test_vecs @ test_vecs.T)
        np.fill_diagonal(wu_cos_matrix, 0)
        wu_cos_off = wu_cos_matrix[mask]
        
        print(f"    W_U行间|cos|: mean={wu_cos_off.mean():.4f}")
        print(f"    logit相关 / W_U|cos| 比值: {np.abs(off_diag).mean() / wu_cos_off.mean():.4f}")
        
        results['global_uniqueness'] = {
            'logit_corr_mean': float(np.abs(off_diag).mean()),
            'logit_corr_std': float(off_diag.std()),
            'logit_corr_max': float(np.abs(off_diag).max()),
            'W_U_cos_mean': float(wu_cos_off.mean()),
            'ratio': float(np.abs(off_diag).mean() / wu_cos_off.mean()) if wu_cos_off.mean() > 0 else 0,
            'conclusion': 'logit独立→全局唯一性成立!' if np.abs(off_diag).mean() < 0.1 else 'logit不独立→需要更深入分析',
        }
    
    print("\n=== P728 完成 ===")
    return results


def P729_orthogonal_intervention(model, tokenizer, device, model_info, model_name):
    """
    P729: 正交干预实验 — 修改风格子空间投影, 观察逻辑/语法logit是否不变
    
    如果风格子空间 ⊥ 逻辑子空间, 则修改风格不影响逻辑 → 正交性→独立性成立!
    """
    print("\n=== P729: 正交干预实验 ===")
    
    W_U = get_W_U(model)
    W_U_np = to_numpy(W_U)
    d = W_U_np.shape[1]
    
    # 定义三类词
    style_words = ['therefore', 'however', 'gonna', 'wanna', 'cool', 'awesome', 
                   'bloom', 'whisper', 'dream', 'eternal', 'yeah', 'nah']
    logic_words = ['because', 'therefore', 'if', 'then', 'since', 'although',
                   'reason', 'cause', 'effect', 'therefore', 'consequently']
    syntax_words = ['the', 'a', 'is', 'are', 'was', 'were', 'has', 'have',
                    'will', 'would', 'can', 'could']
    content_words = ['cat', 'dog', 'house', 'car', 'tree', 'book', 'water', 'fire',
                     'king', 'queen', 'man', 'woman', 'run', 'walk', 'eat', 'think']
    
    # tokenize
    style_ids = tokenize_words(style_words, tokenizer, model_name)
    logic_ids = tokenize_words(logic_words, tokenizer, model_name)
    syntax_ids = tokenize_words(syntax_words, tokenizer, model_name)
    content_ids = tokenize_words(content_words, tokenizer, model_name)
    
    # 获取词向量
    style_vecs = {w: W_U_np[idx] for w, idx in style_ids.items() if idx < W_U_np.shape[0]}
    logic_vecs = {w: W_U_np[idx] for w, idx in logic_ids.items() if idx < W_U_np.shape[0]}
    syntax_vecs = {w: W_U_np[idx] for w, idx in syntax_ids.items() if idx < W_U_np.shape[0]}
    content_vecs = {w: W_U_np[idx] for w, idx in content_ids.items() if idx < W_U_np.shape[0]}
    
    print(f"  找到的词向量数: style={len(style_vecs)}, logic={len(logic_vecs)}, "
          f"syntax={len(syntax_vecs)}, content={len(content_vecs)}")
    
    # 构建子空间
    def build_subspace(vecs_dict, n_comp=5):
        if len(vecs_dict) < 2:
            return None
        vecs = np.array(list(vecs_dict.values()))
        pca = PCA(n_components=min(n_comp, vecs.shape[0]-1, vecs.shape[1]))
        pca.fit(vecs)
        return pca.components_  # [n_comp, d]
    
    style_subspace = build_subspace(style_vecs)
    logic_subspace = build_subspace(logic_vecs)
    syntax_subspace = build_subspace(syntax_vecs)
    
    results = {}
    
    if style_subspace is None or logic_subspace is None:
        print("  子空间构建失败(词太少), 跳过P729")
        return {'error': 'insufficient words for subspace construction'}
    
    # 干预实验: 修改h在风格子空间的投影, 观察各类词logit的变化
    print("\n  干预实验: 增强风格子空间投影...")
    
    # 生成一个"中性"h
    h_base = np.random.randn(d).astype(np.float32)
    h_base = h_base / np.linalg.norm(h_base) * 10  # 归一化到合理范围
    
    # 计算原始logits
    all_test_words = {}
    all_test_words.update({f'style_{w}': v for w, v in style_vecs.items()})
    all_test_words.update({f'logic_{w}': v for w, v in logic_vecs.items()})
    all_test_words.update({f'content_{w}': v for w, v in content_vecs.items()})
    
    original_logits = {}
    for name, vec in all_test_words.items():
        original_logits[name] = float(h_base @ vec)
    
    # 干预: 增强风格子空间投影
    intervention_results = {}
    for scale in [0.5, 1.0, 2.0, 3.0, 5.0]:
        # 投影到风格子空间
        style_proj = style_subspace.T @ (style_subspace @ h_base)  # h在风格子空间的投影
        
        # 修改: h_modified = h_base + (scale-1) * style_proj
        h_modified = h_base + (scale - 1.0) * style_proj
        
        # 计算修改后的logits
        modified_logits = {}
        for name, vec in all_test_words.items():
            modified_logits[name] = float(h_modified @ vec)
        
        # 计算logit变化
        logit_changes = {}
        for name in all_test_words:
            logit_changes[name] = modified_logits[name] - original_logits[name]
        
        # 按类别汇总
        cat_changes = defaultdict(list)
        for name, change in logit_changes.items():
            cat = name.split('_')[0]
            cat_changes[cat].append(change)
        
        cat_mean_change = {cat: np.mean(changes) for cat, changes in cat_changes.items()}
        cat_abs_change = {cat: np.mean(np.abs(changes)) for cat, changes in cat_changes.items()}
        
        intervention_results[f'scale_{scale}'] = {
            'mean_change': {k: float(v) for k, v in cat_mean_change.items()},
            'abs_change': {k: float(v) for k, v in cat_abs_change.items()},
        }
        
        print(f"    scale={scale}: style_change={cat_abs_change.get('style', 0):.4f}, "
              f"logic_change={cat_abs_change.get('logic', 0):.4f}, "
              f"content_change={cat_abs_change.get('content', 0):.4f}")
    
    results['intervention_results'] = intervention_results
    
    # 正交性→独立性判定
    # 如果增强风格投影→风格logit变化大但logic/content变化小 → 正交性→独立性成立
    scale5 = intervention_results.get('scale_5.0', {})
    style_effect = scale5.get('abs_change', {}).get('style', 0)
    logic_effect = scale5.get('abs_change', {}).get('logic', 0)
    content_effect = scale5.get('abs_change', {}).get('content', 0)
    
    if style_effect > 0:
        independence_ratio = {
            'logic_vs_style': logic_effect / style_effect if style_effect > 0 else float('inf'),
            'content_vs_style': content_effect / style_effect if style_effect > 0 else float('inf'),
        }
    else:
        independence_ratio = {'logic_vs_style': 0, 'content_vs_style': 0}
    
    print(f"\n  独立性比值: logic/style={independence_ratio['logic_vs_style']:.3f}, "
          f"content/style={independence_ratio['content_vs_style']:.3f}")
    print(f"  (比值越低→独立性越强→正交性→独立性成立)")
    
    results['independence_ratio'] = {k: float(v) for k, v in independence_ratio.items()}
    results['conclusion'] = (
        '正交性→独立性成立!' if independence_ratio['logic_vs_style'] < 0.5
        else '独立性中等' if independence_ratio['logic_vs_style'] < 1.0
        else '独立性弱→正交子空间可能不存在'
    )
    
    print("\n=== P729 完成 ===")
    return results


def P730_cross_model_orthogonality(model, tokenizer, device, model_info, model_name):
    """
    P730: 跨模型正交性一致性
    - 不同模型的W_U子空间结构是否一致?
    - 旋转对齐: 两个模型的W_U主成分是否可以通过旋转对齐?
    """
    print("\n=== P730: 跨模型正交性一致性 ===")
    print("  (单模型运行, 保存子空间结构供后续跨模型比较)")
    
    W_U = get_W_U(model)
    W_U_np = to_numpy(W_U)
    
    # PCA
    pca = PCA(n_components=min(50, W_U_np.shape[1]))
    pca.fit(W_U_np)
    
    results = {
        'model': model_name,
        'explained_var_top20': pca.explained_variance_ratio_[:20].tolist(),
        'total_var_50pc': float(pca.explained_variance_ratio_[:50].sum()),
        'components_shape': list(pca.components_.shape),
    }
    
    # 保存主成分供跨模型对齐
    save_dir = f'results/phase_clxix/{model_name}'
    os.makedirs(save_dir, exist_ok=True)
    np.save(f'{save_dir}/pc_components.npy', pca.components_[:20])
    np.save(f'{save_dir}/explained_var.npy', pca.explained_variance_ratio_[:20])
    
    # 分析前20个PC的语义
    # 用SYNTAX_CATS等计算各PC的偏好
    word_ids = {}
    for cat_name, words in SYNTAX_CATS.items():
        word_ids.update(tokenize_words(words, tokenizer, model_name))
    
    vectors_dict = get_word_vectors(word_ids, W_U)
    
    if len(vectors_dict) > 10:
        vecs = np.array(list(vectors_dict.values()))
        word_names = list(vectors_dict.keys())
        
        # 标注每个词的语法类别
        word_syntax = {}
        for cat, words in SYNTAX_CATS.items():
            for w in words:
                word_syntax[w] = cat
        
        # 各PC上的投影
        proj = vecs @ pca.components_[:20].T  # [n_words, 20]
        
        # 每个PC: 哪个语法类别的词投影最大
        pc_syntax_pref = []
        for pc_idx in range(20):
            cat_scores = defaultdict(list)
            for i, w in enumerate(word_names):
                if w in word_syntax:
                    cat_scores[word_syntax[w]].append(abs(proj[i, pc_idx]))
            
            cat_mean = {c: np.mean(s) for c, s in cat_scores.items() if len(s) >= 2}
            if cat_mean:
                top_cat = max(cat_mean, key=cat_mean.get)
                pc_syntax_pref.append({
                    'pc': pc_idx,
                    'top_syntax': top_cat,
                    'top_score': float(cat_mean[top_cat]),
                    'all_scores': {k: float(v) for k, v in sorted(cat_mean.items(), key=lambda x: -x[1])[:5]},
                })
        
        results['pc_syntax_preference'] = pc_syntax_pref
    
    print(f"  前20个PC总方差: {results['total_var_50pc']:.4f}")
    print(f"  子空间结构已保存到 {save_dir}/")
    
    print("\n=== P730 完成 ===")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        choices=['qwen3', 'deepseek7b', 'glm4'])
    args = parser.parse_args()
    model_name = args.model
    
    print(f"\n{'='*60}")
    print(f"Phase CLXIX: W_U正交子空间分解 — {model_name}")
    print(f"{'='*60}")
    
    # 加载模型
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    
    # 获取W_U
    W_U = get_W_U(model)
    
    # tokenize所有词
    all_words = set()
    for cat_words in SYNTAX_CATS.values():
        all_words.update(cat_words)
    for cat_words in SEMANTIC_DOMAINS.values():
        all_words.update(cat_words)
    for cat_words in STYLE_MARKERS.values():
        all_words.update(cat_words)
    
    word_ids = tokenize_words(list(all_words), tokenizer, model_name)
    vectors_dict = get_word_vectors(word_ids, W_U)
    
    print(f"\n  总词数: {len(all_words)}, 找到向量: {len(vectors_dict)}")
    
    W_U_np = to_numpy(W_U)
    
    # 运行三个实验
    results = {}
    
    # P728
    try:
        results["P728"] = analyze_subspace_alignment(vectors_dict, W_U_np, n_pc=50)
    except Exception as e:
        print(f"P728失败: {e}")
        import traceback
        traceback.print_exc()
        results["P728"] = {"error": str(e)}
    
    # P729
    try:
        results["P729"] = P729_orthogonal_intervention(model, tokenizer, device, model_info, model_name)
    except Exception as e:
        print(f"P729失败: {e}")
        import traceback
        traceback.print_exc()
        results["P729"] = {"error": str(e)}
    
    # P730
    try:
        results["P730"] = P730_cross_model_orthogonality(model, tokenizer, device, model_info, model_name)
    except Exception as e:
        print(f"P730失败: {e}")
        import traceback
        traceback.print_exc()
        results["P730"] = {"error": str(e)}
    
    # 保存结果
    save_dir = f'results/phase_clxix'
    os.makedirs(save_dir, exist_ok=True)
    save_path = f'{save_dir}/{model_name}_results.json'
    
    # 添加model信息
    results['model_info'] = model_info
    results['model_name'] = model_name
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n结果已保存到: {save_path}")
    
    # 释放GPU
    del model
    import gc
    gc.collect()
    import torch
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
