"""
Phase CLXXI: 正交子空间×频谱力学交叉分析
==========================================
核心目标: 合并两条路线, 在正交子空间内做频谱分析, 破解排序信息

关键假说:
  如果正交子空间(语法/语义/风格)各自有独立的频谱结构,
  那么排序信息 = 子空间间频谱的耦合

实验:
  P734: 各子空间的频谱结构
    - 在语法/语义/风格子空间内分别做SVD
    - 比较各子空间的奇异值谱
    - 假说: 不同子空间有不同的频谱衰减速率(alpha)

  P735: 子空间频谱与词频的关系
    - 低频词vs高频词在各子空间中的频谱分布
    - 假说: 低频词在语义子空间中频谱更集中

  P736: 子空间耦合→排序信息
    - 分析h在各子空间的投影能量比 vs logit排序
    - 假说: 子空间间的能量分配决定了词的排序
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import gc
import numpy as np
import torch
from scipy import stats
from sklearn.decomposition import PCA
from pathlib import Path

from model_utils import load_model, get_model_info, get_W_U, get_layers


def to_numpy(tensor_or_array):
    """统一转换为numpy float32数组"""
    if isinstance(tensor_or_array, np.ndarray):
        return tensor_or_array.astype(np.float32)
    return tensor_or_array.detach().cpu().float().numpy().astype(np.float32)


# ============================================================
# 词分类定义
# ============================================================

# 语法角色分类
SYNTAX_CATS = {
    'noun': ['cat', 'dog', 'house', 'car', 'tree', 'book', 'water', 'food',
             'hand', 'head', 'eye', 'door', 'table', 'chair', 'window',
             'apple', 'sun', 'stone', 'hair', 'fire', 'king', 'queen',
             'man', 'woman', 'boy', 'girl', 'child', 'father', 'mother'],
    'verb': ['run', 'walk', 'eat', 'drink', 'see', 'hear', 'think', 'know',
             'make', 'take', 'give', 'come', 'go', 'sit', 'stand', 'write',
             'read', 'speak', 'listen', 'feel', 'love', 'hate', 'want', 'need',
             'can', 'will', 'should', 'must', 'may', 'might', 'could', 'would'],
    'adj': ['big', 'small', 'good', 'bad', 'hot', 'cold', 'new', 'old',
            'fast', 'slow', 'high', 'low', 'long', 'short', 'strong', 'weak',
            'happy', 'sad', 'beautiful', 'ugly', 'bright', 'dark', 'heavy', 'light'],
    'adv': ['quickly', 'slowly', 'carefully', 'easily', 'always', 'never',
            'often', 'rarely', 'very', 'quite', 'really', 'almost', 'just', 'still'],
    'prep': ['in', 'on', 'at', 'to', 'from', 'with', 'by', 'for', 'of', 'about',
             'between', 'through', 'under', 'over', 'after', 'before'],
    'pron': ['he', 'she', 'it', 'they', 'we', 'you', 'I', 'me', 'him', 'her',
             'them', 'us', 'this', 'that', 'these', 'those', 'my', 'your', 'his'],
    'det': ['the', 'a', 'an', 'this', 'that', 'some', 'any', 'all', 'every',
            'each', 'no', 'both', 'few', 'many', 'much', 'several'],
}

# 语义分类
SEMANTIC_CATS = {
    'animal': ['cat', 'dog', 'bird', 'fish', 'horse', 'cow', 'sheep', 'pig',
               'mouse', 'rat', 'rabbit', 'snake', 'lion', 'tiger', 'bear'],
    'food': ['apple', 'bread', 'rice', 'meat', 'fish', 'cake', 'milk', 'water',
             'tea', 'coffee', 'beer', 'wine', 'soup', 'cheese', 'egg'],
    'color': ['red', 'blue', 'green', 'yellow', 'black', 'white', 'pink',
              'purple', 'orange', 'brown', 'gray', 'gold', 'silver'],
    'emotion': ['love', 'hate', 'joy', 'sad', 'angry', 'fear', 'hope',
                'pride', 'shame', 'guilt', 'envy', 'pity', 'calm'],
    'motion': ['run', 'walk', 'fly', 'swim', 'jump', 'climb', 'fall',
               'rise', 'move', 'stop', 'turn', 'spin', 'slide'],
    'abstract': ['time', 'space', 'truth', 'beauty', 'justice', 'freedom',
                 'power', 'knowledge', 'love', 'death', 'life', 'mind'],
}

# 风格词
STYLE_CATS = {
    'formal': ['therefore', 'consequently', 'furthermore', 'nevertheless',
               'accordingly', 'henceforth', 'whereby', 'herein',
               'notwithstanding', 'hence', 'thus', 'moreover'],
    'informal': ['yeah', 'nah', 'cool', 'wow', 'hey', 'oh', 'um',
                 'like', 'basically', 'literally', 'awesome', 'dude'],
    'literary': ['twilight', 'whisper', 'shadow', 'eternal', 'destiny',
                 'solitude', 'echoes', 'dreams', 'harvest', 'ancient'],
    'technical': ['algorithm', 'parameter', 'function', 'variable', 'optimize',
                  'convergence', 'gradient', 'matrix', 'tensor', 'entropy'],
}


def get_word_ids(cats, tokenizer):
    """从分类字典获取词的token id"""
    word_ids = {}
    for cat, words in cats.items():
        cat_ids = {}
        for w in words:
            tokens = tokenizer.encode(w, add_special_tokens=False)
            if len(tokens) == 1:
                cat_ids[w] = tokens[0]
        word_ids[cat] = cat_ids
    return word_ids


def build_subspace_matrix(word_ids, W_U_np, cat_names=None):
    """构建子空间基矩阵: 收集各类词的W_U行向量"""
    if cat_names is None:
        cat_names = list(word_ids.keys())
    
    vectors = []
    for cat in cat_names:
        if cat in word_ids:
            for w, idx in word_ids[cat].items():
                if idx < W_U_np.shape[0]:
                    vectors.append(W_U_np[idx])
    
    if not vectors:
        return None
    
    V = np.array(vectors)  # [n_words, d]
    # PCA得到子空间基
    pca = PCA(n_components=min(20, V.shape[0], V.shape[1]))
    pca.fit(V)
    return pca.components_  # [n_pc, d]


# ============================================================
# P734: 各子空间的频谱结构
# ============================================================

def P734_subspace_spectral(model, tokenizer, device, model_info, model_name):
    """
    P734: 各子空间的频谱结构
    - 在语法/语义/风格子空间内分别做SVD
    - 比较各子空间的奇异值谱
    - 假说: 不同子空间有不同的频谱衰减速率(alpha)
    """
    print("\n=== P734: 各子空间的频谱结构 ===")
    
    W_U = to_numpy(get_W_U(model))
    n_vocab, d_model = W_U.shape
    
    # 获取各类词的id
    syntax_ids = get_word_ids(SYNTAX_CATS, tokenizer)
    semantic_ids = get_word_ids(SEMANTIC_CATS, tokenizer)
    style_ids = get_word_ids(STYLE_CATS, tokenizer)
    
    # 构建各子空间基
    print("  构建子空间基...")
    syntax_basis = build_subspace_matrix(syntax_ids, W_U)
    semantic_basis = build_subspace_matrix(semantic_ids, W_U)
    style_basis = build_subspace_matrix(style_ids, W_U)
    
    results = {}
    
    # 对每个子空间做SVD分析
    for name, basis in [('syntax', syntax_basis), ('semantic', semantic_basis), ('style', style_basis)]:
        if basis is None:
            results[name] = {'error': 'no basis'}
            continue
        
        print(f"\n  分析 {name} 子空间频谱...")
        
        # basis: [n_pc, d], 每行是一个主成分方向
        # SVD of basis
        U, S, Vt = np.linalg.svd(basis, full_matrices=False)
        
        # 奇异值谱
        S_norm = S / S[0]  # 归一化
        
        # 幂律拟合: S_k ~ k^(-alpha)
        log_k = np.log(np.arange(1, len(S)+1))
        log_S = np.log(S_norm + 1e-10)
        
        # 线性拟合 log(S) = -alpha * log(k) + c
        valid = S_norm > 1e-6
        if valid.sum() > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                log_k[valid], log_S[valid])
            alpha = -slope
            r_squared = r_value**2
        else:
            alpha, r_squared, p_value = 0, 0, 1
        
        # 将W_U投影到子空间, 分析投影的频谱
        proj = W_U @ basis.T  # [n_vocab, n_pc]
        
        # 投影的奇异值谱
        U_proj, S_proj, Vt_proj = np.linalg.svd(proj, full_matrices=False)
        S_proj_norm = S_proj / S_proj[0]
        
        # 幂律拟合投影谱
        log_S_proj = np.log(S_proj_norm[:min(50, len(S_proj_norm))] + 1e-10)
        log_k_proj = np.log(np.arange(1, len(log_S_proj)+1))
        valid_proj = S_proj_norm[:len(log_S_proj)] > 1e-6
        if valid_proj.sum() > 2:
            slope_proj, intercept_proj, r_proj, p_proj, _ = stats.linregress(
                log_k_proj[valid_proj], log_S_proj[valid_proj])
            alpha_proj = -slope_proj
            r_squared_proj = r_proj**2
        else:
            alpha_proj, r_squared_proj, p_proj = 0, 0, 1
        
        # 子空间内的能量分布
        proj_energy = np.sum(proj**2, axis=0)  # [n_pc]
        total_energy = np.sum(proj_energy)
        energy_ratio = proj_energy / total_energy if total_energy > 0 else proj_energy * 0
        
        # top-5和top-10能量集中度
        top5_energy = np.sum(energy_ratio[:5])
        top10_energy = np.sum(energy_ratio[:10])
        
        # 子空间维度(有效维度)
        effective_dim = 1.0 / np.sum(energy_ratio**2) if np.sum(energy_ratio**2) > 0 else 0
        
        results[name] = {
            'n_basis_vectors': basis.shape[0],
            'basis_singular_values': S_norm.tolist()[:20],
            'basis_alpha': float(alpha),
            'basis_r_squared': float(r_squared),
            'basis_p_value': float(p_value),
            'proj_alpha': float(alpha_proj),
            'proj_r_squared': float(r_squared_proj),
            'proj_top5_energy': float(top5_energy),
            'proj_top10_energy': float(top10_energy),
            'effective_dim': float(effective_dim),
            'energy_distribution': energy_ratio.tolist()[:20],
        }
        
        print(f"    基频谱alpha={alpha:.3f}(R2={r_squared:.3f})")
        print(f"    投影频谱alpha={alpha_proj:.3f}(R2={r_squared_proj:.3f})")
        print(f"    Top-5能量集中度={top5_energy:.3f}, Top-10={top10_energy:.3f}")
        print(f"    有效维度={effective_dim:.1f}")
    
    # 子空间间频谱比较
    print("\n  子空间频谱比较:")
    alphas = {}
    for name in ['syntax', 'semantic', 'style']:
        if name in results and 'error' not in results[name]:
            alphas[name] = results[name]['proj_alpha']
    
    if len(alphas) >= 2:
        print(f"    频谱衰减率(alpha): {alphas}")
        max_alpha_name = max(alphas, key=alphas.get)
        min_alpha_name = min(alphas, key=alphas.get)
        print(f"    最快衰减: {max_alpha_name}(alpha={alphas[max_alpha_name]:.3f})")
        print(f"    最慢衰减: {min_alpha_name}(alpha={alphas[min_alpha_name]:.3f})")
    
    # 关键分析: 各子空间是否使用不同的频段?
    print("\n  子空间频段偏好分析:")
    for name in ['syntax', 'semantic', 'style']:
        if name in results and 'error' not in results[name]:
            energy = results[name]['energy_distribution']
            if len(energy) >= 10:
                low_band = sum(energy[:5])   # PC 0-4: 低频段
                high_band = sum(energy[5:10]) # PC 5-9: 高频段
                print(f"    {name}: 低频段={low_band:.3f}, 高频段={high_band:.3f}, 比={low_band/(high_band+1e-10):.2f}")
    
    results['conclusion'] = '不同子空间有不同频谱衰减率' if len(set(alphas.values())) > 1 else '所有子空间频谱衰减率相同'
    
    print("\n=== P734 完成 ===")
    return results


# ============================================================
# P735: 子空间频谱与词频的关系
# ============================================================

def P735_word_freq_spectral(model, tokenizer, device, model_info, model_name):
    """
    P735: 子空间频谱与词频的关系
    - 分析高频词vs低频词在各子空间中的频谱分布
    - 假说: 低频词在语义子空间中频谱更集中
    """
    print("\n=== P735: 子空间频谱与词频的关系 ===")
    
    W_U = to_numpy(get_W_U(model))
    n_vocab, d_model = W_U.shape
    
    # 构建子空间基
    syntax_ids = get_word_ids(SYNTAX_CATS, tokenizer)
    semantic_ids = get_word_ids(SEMANTIC_CATS, tokenizer)
    style_ids = get_word_ids(STYLE_CATS, tokenizer)
    
    syntax_basis = build_subspace_matrix(syntax_ids, W_U)
    semantic_basis = build_subspace_matrix(semantic_ids, W_U)
    style_basis = build_subspace_matrix(style_ids, W_U)
    
    # 定义高频词和低频词(使用常见/罕见英语词)
    HIGH_FREQ_WORDS = ['the', 'a', 'is', 'are', 'was', 'were', 'be', 'been',
                       'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                       'can', 'could', 'may', 'might', 'should', 'must', 'shall',
                       'not', 'no', 'but', 'and', 'or', 'if', 'so', 'as', 'at',
                       'by', 'for', 'from', 'in', 'into', 'of', 'on', 'to', 'with']
    
    LOW_FREQ_WORDS = ['ephemeral', 'ubiquitous', 'paradigm', 'synergy', 'entropy',
                      'quantum', 'neural', 'catalyst', 'dichotomy', 'paradox',
                      'metamorphosis', 'serendipity', 'eloquent', 'cacophony',
                      'labyrinth', 'zenith', 'nadir', 'juxtaposition', 'ephemera',
                      'quintessential', 'surreptitious', 'magnanimous', 'pulchritude',
                      'sesquipedalian', 'perspicacious', 'obfuscate', 'conflagration',
                      'ineffable', 'luminous', 'resplendent', 'verisimilitude']
    
    # 获取词向量
    def get_vectors(words):
        vecs = {}
        for w in words:
            tokens = tokenizer.encode(w, add_special_tokens=False)
            if len(tokens) == 1 and tokens[0] < W_U.shape[0]:
                vecs[w] = W_U[tokens[0]]
        return vecs
    
    high_freq_vecs = get_vectors(HIGH_FREQ_WORDS)
    low_freq_vecs = get_vectors(LOW_FREQ_WORDS)
    
    results = {}
    
    # 分析各子空间
    for subspace_name, basis in [('syntax', syntax_basis), ('semantic', semantic_basis), ('style', style_basis)]:
        if basis is None:
            continue
        
        print(f"\n  分析 {subspace_name} 子空间中的词频差异...")
        
        # 投影高频词和低频词到子空间
        high_projs = []
        for w, v in high_freq_vecs.items():
            proj = v @ basis.T  # [n_pc]
            high_projs.append(proj)
        
        low_projs = []
        for w, v in low_freq_vecs.items():
            proj = v @ basis.T  # [n_pc]
            low_projs.append(proj)
        
        if not high_projs or not low_projs:
            continue
        
        high_projs = np.array(high_projs)  # [n_high, n_pc]
        low_projs = np.array(low_projs)    # [n_low, n_pc]
        
        # 各频段的能量
        n_pc = min(high_projs.shape[1], low_projs.shape[1], 20)
        
        # 低频段(PC 0-4)和高频段(PC 5-9)的能量
        low_band_idx = min(5, n_pc)
        high_band_idx = min(10, n_pc)
        
        high_low_band = np.mean(np.sum(high_projs[:, :low_band_idx]**2, axis=1))
        high_high_band = np.mean(np.sum(high_projs[:, low_band_idx:high_band_idx]**2, axis=1)) if high_band_idx > low_band_idx else 0
        high_total = np.mean(np.sum(high_projs[:, :n_pc]**2, axis=1))
        
        low_low_band = np.mean(np.sum(low_projs[:, :low_band_idx]**2, axis=1))
        low_high_band = np.mean(np.sum(low_projs[:, low_band_idx:high_band_idx]**2, axis=1)) if high_band_idx > low_band_idx else 0
        low_total = np.mean(np.sum(low_projs[:, :n_pc]**2, axis=1))
        
        # 集中度: top-5能量/总能量
        high_concentration = high_low_band / (high_total + 1e-10)
        low_concentration = low_low_band / (low_total + 1e-10)
        
        # 统计检验
        high_conc_per_word = np.sum(high_projs[:, :low_band_idx]**2, axis=1) / (np.sum(high_projs[:, :n_pc]**2, axis=1) + 1e-10)
        low_conc_per_word = np.sum(low_projs[:, :low_band_idx]**2, axis=1) / (np.sum(low_projs[:, :n_pc]**2, axis=1) + 1e-10)
        
        if len(high_conc_per_word) > 1 and len(low_conc_per_word) > 1:
            t_stat, p_value = stats.ttest_ind(high_conc_per_word, low_conc_per_word)
        else:
            t_stat, p_value = 0, 1
        
        results[subspace_name] = {
            'n_high_freq': len(high_projs),
            'n_low_freq': len(low_projs),
            'high_freq_low_band_energy': float(high_low_band),
            'high_freq_high_band_energy': float(high_high_band),
            'high_freq_total_energy': float(high_total),
            'high_freq_concentration': float(high_concentration),
            'low_freq_low_band_energy': float(low_low_band),
            'low_freq_high_band_energy': float(low_high_band),
            'low_freq_total_energy': float(low_total),
            'low_freq_concentration': float(low_concentration),
            'concentration_t_stat': float(t_stat),
            'concentration_p_value': float(p_value),
        }
        
        print(f"    高频词: 总能量={high_total:.2f}, 集中度={high_concentration:.4f}")
        print(f"    低频词: 总能量={low_total:.2f}, 集中度={low_concentration:.4f}")
        print(f"    差异: t={t_stat:.3f}, p={p_value:.4f}")
    
    # 总结
    print("\n  词频x子空间总结:")
    for name in ['syntax', 'semantic', 'style']:
        if name in results:
            r = results[name]
            conc_diff = r['low_freq_concentration'] - r['high_freq_concentration']
            direction = '低频词更集中' if conc_diff > 0 else '高频词更集中'
            sig = '***' if r['concentration_p_value'] < 0.001 else '**' if r['concentration_p_value'] < 0.01 else '*' if r['concentration_p_value'] < 0.05 else 'n.s.'
            print(f"    {name}: {direction} (diff={conc_diff:.4f}, {sig})")
    
    results['conclusion'] = '低频词在语义子空间中频谱更集中' if results.get('semantic', {}).get('concentration_p_value', 1) < 0.05 else '词频与子空间频谱集中度无显著关系'
    
    print("\n=== P735 完成 ===")
    return results


# ============================================================
# P736: 子空间耦合→排序信息
# ============================================================

def P736_subspace_coupling_ranking(model, tokenizer, device, model_info, model_name):
    """
    P736: 子空间耦合→排序信息
    - 分析h在各子空间的投影能量比 vs logit排序
    - 假说: 子空间间的能量分配决定了词的排序
    - 这是两条路线合并的核心实验!
    """
    print("\n=== P736: 子空间耦合→排序信息 ===")
    
    W_U = to_numpy(get_W_U(model))
    n_vocab, d_model = W_U.shape
    n_layers = model_info.n_layers
    
    # 构建子空间基
    syntax_ids = get_word_ids(SYNTAX_CATS, tokenizer)
    semantic_ids = get_word_ids(SEMANTIC_CATS, tokenizer)
    style_ids = get_word_ids(STYLE_CATS, tokenizer)
    
    syntax_basis = build_subspace_matrix(syntax_ids, W_U)
    semantic_basis = build_subspace_matrix(semantic_ids, W_U)
    style_basis = build_subspace_matrix(style_ids, W_U)
    
    # 定义测试句子
    TEST_SENTENCES = [
        "The cat sat on the",
        "She walked to the",
        "He said that the",
        "The weather is very",
        "I think that the",
        "The book was about",
        "They went to the",
        "She looked at the",
        "The man who was",
        "It is important to",
    ]
    
    results = {}
    
    # 获取各层残差流
    layers = get_layers(model)
    
    for sent_idx, sent in enumerate(TEST_SENTENCES):
        print(f"\n  分析句子: '{sent}'")
        
        tokens = tokenizer.encode(sent, return_tensors='pt').to(device)
        
        # 注册hook
        layer_outputs = {}
        hooks = []
        
        def make_hook(lidx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden = output[0]
                else:
                    hidden = output
                layer_outputs[lidx] = hidden.detach()
            return hook_fn
        
        for layer_idx, layer in enumerate(layers[:n_layers]):
            h = layer.register_forward_hook(make_hook(layer_idx))
            hooks.append(h)
        
        with torch.no_grad():
            try:
                outputs = model(tokens)
            except Exception as e:
                print(f"    前向传播失败: {e}")
                for h in hooks:
                    h.remove()
                continue
        
        for h in hooks:
            h.remove()
        
        # 获取最后一层的logits和h
        if not layer_outputs:
            continue
        
        last_layer_idx = max(layer_outputs.keys())
        last_pos = tokens.shape[1] - 1
        
        if len(layer_outputs[last_layer_idx].shape) == 3:
            h_final = to_numpy(layer_outputs[last_layer_idx][0, last_pos])  # [d_model]
        else:
            continue
        
        # 计算logits
        logits = h_final @ W_U.T  # [n_vocab]
        
        # Top-20词
        top20_idx = np.argsort(logits)[-20:][::-1]
        top20_words = [tokenizer.decode([idx]) for idx in top20_idx]
        top20_logits = logits[top20_idx]
        
        # 分析: Top-20词在各子空间的投影能量
        # h_final投影到各子空间
        subspaces = {}
        if syntax_basis is not None:
            subspaces['syntax'] = syntax_basis
        if semantic_basis is not None:
            subspaces['semantic'] = semantic_basis
        if style_basis is not None:
            subspaces['style'] = style_basis
        
        # h的子空间投影
        h_proj = {}
        for sname, basis in subspaces.items():
            h_proj[sname] = h_final @ basis.T  # [n_pc]
        
        # 各子空间的能量
        h_energy = {}
        total_h_energy = np.sum(h_final**2)
        for sname, proj in h_proj.items():
            h_energy[sname] = np.sum(proj**2) / total_h_energy
        
        # 关键分析: 每个top词的logit分解
        # logit_i = h @ W_U[i] = sum_subspace (h @ P_subspace) @ W_U[i]
        #         = sum_subspace contribution_subspace_i
        
        word_decompositions = {}
        for rank, word_idx in enumerate(top20_idx[:10]):
            w_vec = W_U[word_idx]  # [d_model]
            
            decomposition = {}
            total_logit = 0
            for sname, basis in subspaces.items():
                # 子空间贡献 = (h在子空间投影) @ (w_vec在子空间投影)
                h_in_sub = h_final @ basis.T @ basis  # h投影回原空间
                contrib = np.dot(h_in_sub, w_vec)
                decomposition[sname] = float(contrib)
                total_logit += contrib
            
            # 残差空间贡献
            h_in_all_subs = np.zeros_like(h_final)
            for sname, basis in subspaces.items():
                h_in_all_subs += h_final @ basis.T @ basis
            residual_contrib = float(np.dot(h_final - h_in_all_subs, w_vec))
            decomposition['residual'] = residual_contrib
            
            word_decompositions[f"rank{rank}_{tokenizer.decode([word_idx]).strip()}"] = decomposition
        
        results[f'sentence_{sent_idx}'] = {
            'text': sent,
            'top5_words': top20_words[:5],
            'top5_logits': top20_logits[:5].tolist(),
            'h_subspace_energy': h_energy,
            'word_decompositions': word_decompositions,
        }
        
        # 释放
        del layer_outputs
        gc.collect()
        torch.cuda.empty_cache()
    
    # 汇总分析: 各子空间对排序的贡献
    print("\n  汇总: 子空间对logit排序的贡献")
    
    # 收集所有词的子空间贡献
    all_syntax_contribs = []
    all_semantic_contribs = []
    all_style_contribs = []
    all_residual_contribs = []
    
    for skey, sdata in results.items():
        if 'word_decompositions' not in sdata:
            continue
        for wkey, decomp in sdata['word_decompositions'].items():
            if 'syntax' in decomp:
                all_syntax_contribs.append(decomp['syntax'])
            if 'semantic' in decomp:
                all_semantic_contribs.append(decomp['semantic'])
            if 'style' in decomp:
                all_style_contribs.append(decomp['style'])
            if 'residual' in decomp:
                all_residual_contribs.append(decomp['residual'])
    
    if all_syntax_contribs and all_semantic_contribs:
        avg_syntax = np.mean(np.abs(all_syntax_contribs))
        avg_semantic = np.mean(np.abs(all_semantic_contribs))
        avg_style = np.mean(np.abs(all_style_contribs)) if all_style_contribs else 0
        avg_residual = np.mean(np.abs(all_residual_contribs)) if all_residual_contribs else 0
        
        total_avg = avg_syntax + avg_semantic + avg_style + avg_residual
        
        results['subspace_contribution_ratio'] = {
            'syntax': float(avg_syntax / total_avg) if total_avg > 0 else 0,
            'semantic': float(avg_semantic / total_avg) if total_avg > 0 else 0,
            'style': float(avg_style / total_avg) if total_avg > 0 else 0,
            'residual': float(avg_residual / total_avg) if total_avg > 0 else 0,
        }
        
        print(f"    语法子空间贡献: {avg_syntax/total_avg*100:.1f}%")
        print(f"    语义子空间贡献: {avg_semantic/total_avg*100:.1f}%")
        print(f"    风格子空间贡献: {avg_style/total_avg*100:.1f}%")
        print(f"    残差空间贡献:   {avg_residual/total_avg*100:.1f}%")
        
        # 关键: 子空间贡献是否区分不同词?
        # 如果语法/语义/风格的贡献在不同词间有不同比例, 那么子空间耦合决定了排序
        if len(all_syntax_contribs) > 5:
            # 计算各子空间贡献的变异系数
            cv_syntax = np.std(all_syntax_contribs) / (np.mean(np.abs(all_syntax_contribs)) + 1e-10)
            cv_semantic = np.std(all_semantic_contribs) / (np.mean(np.abs(all_semantic_contribs)) + 1e-10)
            cv_style = np.std(all_style_contribs) / (np.mean(np.abs(all_style_contribs)) + 1e-10) if all_style_contribs else 0
            
            results['contribution_cv'] = {
                'syntax': float(cv_syntax),
                'semantic': float(cv_semantic),
                'style': float(cv_style),
            }
            
            max_cv_name = max([('syntax', cv_syntax), ('semantic', cv_semantic), ('style', cv_style)], key=lambda x: x[1])
            print(f"    最大变异子空间: {max_cv_name[0]}(CV={max_cv_name[1]:.3f})")
            print(f"    -> 该子空间最决定词的排序")
    
    # h的子空间能量分配 vs 上下文
    print("\n  h的子空间能量分配:")
    for skey, sdata in results.items():
        if 'h_subspace_energy' in sdata:
            energies = sdata['h_subspace_energy']
            total = sum(energies.values())
            print(f"    '{sdata['text'][:30]}': 语法={energies.get('syntax',0)/total*100:.1f}%, "
                  f"语义={energies.get('semantic',0)/total*100:.1f}%, "
                  f"风格={energies.get('style',0)/total*100:.1f}%")
    
    results['conclusion'] = '子空间耦合决定排序' if results.get('contribution_cv', {}).get('semantic', 0) > 0.5 else '子空间耦合与排序关系待验证'
    
    print("\n=== P736 完成 ===")
    return results


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        choices=['qwen3', 'deepseek7b', 'glm4'])
    args = parser.parse_args()
    model_name = args.model
    
    print(f"\n{'='*60}")
    print(f"Phase CLXXI: 正交子空间 x 频谱力学交叉分析 -- {model_name}")
    print(f"{'='*60}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    
    results = {}
    
    # P734
    try:
        results["P734"] = P734_subspace_spectral(model, tokenizer, device, model_info, model_name)
    except Exception as e:
        print(f"P734失败: {e}")
        import traceback
        traceback.print_exc()
        results["P734"] = {"error": str(e)}
    
    # P735
    try:
        results["P735"] = P735_word_freq_spectral(model, tokenizer, device, model_info, model_name)
    except Exception as e:
        print(f"P735失败: {e}")
        import traceback
        traceback.print_exc()
        results["P735"] = {"error": str(e)}
    
    # P736
    try:
        results["P736"] = P736_subspace_coupling_ranking(model, tokenizer, device, model_info, model_name)
    except Exception as e:
        print(f"P736失败: {e}")
        import traceback
        traceback.print_exc()
        results["P736"] = {"error": str(e)}
    
    # 释放模型
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    # 保存结果
    output_dir = Path(f"results/phase_clxxi")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_name}_results.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n结果已保存到: {output_file}")


if __name__ == '__main__':
    main()
