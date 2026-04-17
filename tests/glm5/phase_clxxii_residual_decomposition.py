"""
Phase CLXXII: 残差空间分解 — 逼近完整编码机制
=============================================
核心目标: 分解残差空间(84%的logit贡献), 找到更多有意义的子空间

关键假说:
  当前3个子空间(语法/语义/风格)只捕获了W_U编码的一小部分
  残差空间包含更多子空间, 如: 主题/情感/时态/数量/关系等
  完整分解后, 残差应该趋近0

实验:
  P737: 残差空间的PCA分解
    - 移除语法/语义/风格子空间后, 对残差做PCA
    - 看残差的PCA有哪些有意义的方向
    - 假说: 残差的前几个PC对应新的语言维度

  P738: 残差子空间的语义解码
    - 分析残差PC与词汇属性的关联
    - 假说: 残差PC编码了主题/情感/时态等

  P739: 完整子空间分解 -> 排序公式
    - 用所有子空间(语法+语义+风格+新子空间)重建logit
    - 目标: 排序重建准确率 > 50%
    - 这将是编码机制的完整数学公式!
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
    if isinstance(tensor_or_array, np.ndarray):
        return tensor_or_array.astype(np.float32)
    return tensor_or_array.detach().cpu().float().numpy().astype(np.float32)


# ============================================================
# 词分类定义
# ============================================================

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

# 新增: 更多语言维度
TENSE_CATS = {
    'past': ['was', 'were', 'had', 'did', 'went', 'came', 'saw', 'knew',
             'took', 'gave', 'made', 'said', 'thought', 'felt', 'ran'],
    'present': ['is', 'are', 'has', 'does', 'goes', 'comes', 'sees', 'knows',
                'takes', 'gives', 'makes', 'says', 'thinks', 'feels', 'runs'],
    'future': ['will', 'shall', 'would', 'could', 'should', 'might', 'may',
               'going', 'about', 'intend', 'plan', 'expect'],
}

NUMBER_CATS = {
    'singular': ['cat', 'dog', 'house', 'car', 'tree', 'book', 'man', 'woman',
                 'child', 'person', 'thing', 'place', 'day', 'year', 'hand'],
    'plural': ['cats', 'dogs', 'houses', 'cars', 'trees', 'books', 'men', 'women',
               'children', 'people', 'things', 'places', 'days', 'years', 'hands'],
}

POLARITY_CATS = {
    'positive': ['good', 'great', 'happy', 'love', 'beautiful', 'bright',
                 'warm', 'kind', 'gentle', 'strong', 'fast', 'smart',
                 'hope', 'joy', 'peace', 'success', 'win', 'best'],
    'negative': ['bad', 'terrible', 'sad', 'hate', 'ugly', 'dark',
                 'cold', 'cruel', 'harsh', 'weak', 'slow', 'stupid',
                 'fear', 'anger', 'war', 'failure', 'lose', 'worst'],
}

TOPIC_CATS = {
    'nature': ['tree', 'river', 'mountain', 'ocean', 'forest', 'flower',
               'rain', 'snow', 'wind', 'sun', 'moon', 'star', 'earth', 'sky'],
    'technology': ['computer', 'phone', 'internet', 'software', 'data',
                   'network', 'digital', 'code', 'program', 'system', 'machine'],
    'society': ['government', 'law', 'politics', 'economy', 'culture',
                'education', 'health', 'science', 'religion', 'family'],
    'person': ['man', 'woman', 'child', 'friend', 'teacher', 'doctor',
               'leader', 'worker', 'artist', 'writer', 'soldier', 'mother'],
}


def get_word_ids(cats, tokenizer):
    word_ids = {}
    for cat, words in cats.items():
        cat_ids = {}
        for w in words:
            tokens = tokenizer.encode(w, add_special_tokens=False)
            if len(tokens) == 1:
                cat_ids[w] = tokens[0]
        word_ids[cat] = cat_ids
    return word_ids


def build_subspace_basis(word_ids, W_U_np, n_pc=20):
    vectors = []
    for cat in word_ids:
        for w, idx in word_ids[cat].items():
            if idx < W_U_np.shape[0]:
                vectors.append(W_U_np[idx])
    if not vectors:
        return None
    V = np.array(vectors)
    pca = PCA(n_components=min(n_pc, V.shape[0], V.shape[1]))
    pca.fit(V)
    return pca.components_


def project_onto_subspace(v, basis):
    """将v投影到basis定义的子空间"""
    return basis.T @ (basis @ v)


def remove_subspace(v, basis_list):
    """从v中移除所有子空间的投影, 返回残差"""
    residual = v.copy()
    for basis in basis_list:
        proj = project_onto_subspace(residual, basis)
        residual -= proj
    return residual


# ============================================================
# P737: 残差空间的PCA分解
# ============================================================

def P737_residual_pca(model, tokenizer, device, model_info, model_name):
    """
    P737: 残差空间的PCA分解
    - 移除语法/语义/风格子空间后, 对残差做PCA
    - 看残差的PCA有哪些有意义的方向
    """
    print("\n=== P737: 残差空间的PCA分解 ===")
    
    W_U = to_numpy(get_W_U(model))
    n_vocab, d_model = W_U.shape
    
    # 构建已知子空间基
    syntax_ids = get_word_ids(SYNTAX_CATS, tokenizer)
    semantic_ids = get_word_ids(SEMANTIC_CATS, tokenizer)
    style_ids = get_word_ids(STYLE_CATS, tokenizer)
    
    syntax_basis = build_subspace_basis(syntax_ids, W_U)
    semantic_basis = build_subspace_basis(semantic_ids, W_U)
    style_basis = build_subspace_basis(style_ids, W_U)
    
    known_bases = [b for b in [syntax_basis, semantic_basis, style_basis] if b is not None]
    
    # 计算W_U的残差(移除已知子空间)
    print("  移除已知子空间, 计算残差...")
    W_U_residual = np.zeros_like(W_U)
    for i in range(n_vocab):
        W_U_residual[i] = remove_subspace(W_U[i], known_bases)
    
    # 残差的范数
    residual_norms = np.linalg.norm(W_U_residual, axis=1)
    original_norms = np.linalg.norm(W_U, axis=1)
    residual_ratio = np.mean(residual_norms**2) / np.mean(original_norms**2)
    
    print(f"  残差能量占比: {residual_ratio*100:.1f}%")
    
    # 对残差做PCA
    print("  对残差做PCA...")
    n_pc = min(100, W_U_residual.shape[1])
    pca = PCA(n_components=n_pc)
    pca.fit(W_U_residual)
    
    # 分析残差PCA的奇异值谱
    S = pca.explained_variance_ratio_
    
    # 幂律拟合
    log_k = np.log(np.arange(1, len(S)+1))
    log_S = np.log(S + 1e-10)
    valid = S > 1e-8
    if valid.sum() > 5:
        slope, intercept, r_value, p_value, _ = stats.linregress(log_k[valid], log_S[valid])
        alpha = -slope
        r_squared = r_value**2
    else:
        alpha, r_squared, p_value = 0, 0, 1
    
    print(f"  残差频谱alpha={alpha:.3f}(R2={r_squared:.3f})")
    
    # 分析残差PC方向上的词分布
    print("\n  残差PC方向上的Top词:")
    pc_top_words = {}
    residual_components = pca.components_  # [n_pc, d]
    
    for pc_idx in range(min(10, n_pc)):
        # 计算每个词在残差PC方向上的投影
        proj = W_U_residual @ residual_components[pc_idx]  # [n_vocab]
        
        # Top正/负词
        top_pos_idx = np.argsort(proj)[-5:][::-1]
        top_neg_idx = np.argsort(proj)[:5]
        
        top_pos_words = [tokenizer.decode([i]).strip().encode('ascii', 'replace').decode() for i in top_pos_idx]
        top_neg_words = [tokenizer.decode([i]).strip().encode('ascii', 'replace').decode() for i in top_neg_idx]
        
        pc_top_words[f'PC{pc_idx}'] = {
            'variance_ratio': float(S[pc_idx]),
            'top_positive': top_pos_words,
            'top_negative': top_neg_words,
        }
        
        print(f"    PC{pc_idx} (var={S[pc_idx]*100:.2f}%): +{top_pos_words[:3]} / -{top_neg_words[:3]}")
    
    # 分析残差PC是否与新的语言维度对齐
    print("\n  残差PC与语言维度的对齐:")
    
    new_cats = {
        'tense': TENSE_CATS,
        'number': NUMBER_CATS,
        'polarity': POLARITY_CATS,
        'topic': TOPIC_CATS,
    }
    
    new_subspace_alignment = {}
    for dim_name, cats in new_cats.items():
        dim_ids = get_word_ids(cats, tokenizer)
        dim_basis = build_subspace_basis(dim_ids, W_U)
        if dim_basis is None:
            continue
        
        # 计算残差PC与该维度子空间的余弦相似度
        alignment = {}
        for pc_idx in range(min(10, n_pc)):
            cos_sims = []
            for b in dim_basis:
                cos = np.dot(residual_components[pc_idx], b) / (
                    np.linalg.norm(residual_components[pc_idx]) * np.linalg.norm(b) + 1e-10)
                cos_sims.append(abs(cos))
            max_align = max(cos_sims)
            alignment[f'PC{pc_idx}'] = float(max_align)
        
        # 找到最对齐的PC
        best_pc = max(alignment, key=alignment.get)
        best_align = alignment[best_pc]
        
        new_subspace_alignment[dim_name] = {
            'best_pc': best_pc,
            'best_alignment': float(best_align),
            'pc_alignments': alignment,
        }
        
        print(f"    {dim_name}: 最佳对齐={best_pc}(cos={best_align:.4f})")
    
    results = {
        'residual_energy_ratio': float(residual_ratio),
        'residual_alpha': float(alpha),
        'residual_r_squared': float(r_squared),
        'cumulative_variance_top10': float(np.sum(S[:10])),
        'cumulative_variance_top20': float(np.sum(S[:20])),
        'cumulative_variance_top50': float(np.sum(S[:50])),
        'pc_top_words': pc_top_words,
        'new_subspace_alignment': new_subspace_alignment,
    }
    
    print("\n=== P737 完成 ===")
    return results


# ============================================================
# P738: 残差子空间的语义解码
# ============================================================

def P738_residual_semantic(model, tokenizer, device, model_info, model_name):
    """
    P738: 残差子空间的语义解码
    - 构建新的子空间(时态/数量/极性/主题), 分析它们在残差中的位置
    - 用新子空间+已知子空间重建logit
    """
    print("\n=== P738: 残差子空间的语义解码 ===")
    
    W_U = to_numpy(get_W_U(model))
    n_vocab, d_model = W_U.shape
    
    # 构建所有子空间基
    all_cats = {
        'syntax': SYNTAX_CATS,
        'semantic': SEMANTIC_CATS,
        'style': STYLE_CATS,
        'tense': TENSE_CATS,
        'number': NUMBER_CATS,
        'polarity': POLARITY_CATS,
        'topic': TOPIC_CATS,
    }
    
    all_bases = {}
    all_ids = {}
    for dim_name, cats in all_cats.items():
        ids = get_word_ids(cats, tokenizer)
        basis = build_subspace_basis(ids, W_U)
        if basis is not None:
            all_bases[dim_name] = basis
            all_ids[dim_name] = ids
            print(f"  {dim_name}: {basis.shape[0]} PCs")
    
    # 逐步移除子空间, 计算剩余能量
    print("\n  逐步子空间移除:")
    removal_order = ['syntax', 'semantic', 'style', 'tense', 'number', 'polarity', 'topic']
    cumulative_removed = 0
    
    W_U_current = W_U.copy()
    original_energy = np.mean(np.sum(W_U**2, axis=1))
    
    removal_results = {}
    for dim_name in removal_order:
        if dim_name not in all_bases:
            continue
        
        # 计算当前W_U在该子空间的能量
        proj_energy = 0
        for i in range(n_vocab):
            proj = project_onto_subspace(W_U_current[i], all_bases[dim_name])
            proj_energy += np.sum(proj**2)
        proj_energy /= n_vocab
        
        # 移除
        for i in range(n_vocab):
            W_U_current[i] -= project_onto_subspace(W_U_current[i], all_bases[dim_name])
        
        remaining_energy = np.mean(np.sum(W_U_current**2, axis=1))
        removed_ratio = 1 - remaining_energy / original_energy
        
        removal_results[dim_name] = {
            'subspace_energy': float(proj_energy),
            'cumulative_removed': float(removed_ratio),
            'remaining_ratio': float(remaining_energy / original_energy),
        }
        
        print(f"    {dim_name}: 子空间能量={proj_energy:.2f}, 累计移除={removed_ratio*100:.1f}%, "
              f"剩余={remaining_energy/original_energy*100:.1f}%")
    
    # 关键分析: 各子空间的独立性
    # 如果子空间不正交, 移除顺序会影响结果
    print("\n  子空间间余弦相似度:")
    dim_names = list(all_bases.keys())
    pairwise_cos = {}
    for i in range(len(dim_names)):
        for j in range(i+1, len(dim_names)):
            n1, n2 = dim_names[i], dim_names[j]
            # 计算子空间间的典型余弦
            cos_sims = []
            for b1 in all_bases[n1][:5]:
                for b2 in all_bases[n2][:5]:
                    cos = abs(np.dot(b1, b2) / (np.linalg.norm(b1) * np.linalg.norm(b2) + 1e-10))
                    cos_sims.append(cos)
            mean_cos = np.mean(cos_sims)
            pairwise_cos[f'{n1}_vs_{n2}'] = float(mean_cos)
            if mean_cos > 0.1:
                print(f"    {n1} vs {n2}: cos={mean_cos:.4f} (!有重叠)")
    
    # 分析新子空间的秩结构
    print("\n  新子空间秩结构:")
    for dim_name in ['tense', 'number', 'polarity', 'topic']:
        if dim_name not in all_bases:
            continue
        # 投影W_U到子空间
        proj = W_U @ all_bases[dim_name].T  # [n_vocab, n_pc]
        proj_energy = np.sum(proj**2, axis=0)
        total_e = np.sum(proj_energy)
        if total_e > 0:
            energy_ratio = proj_energy / total_e
            effective_dim = 1.0 / np.sum(energy_ratio**2)
            top1_ratio = float(energy_ratio[0])
            print(f"    {dim_name}: 有效维度={effective_dim:.1f}, Top1={top1_ratio*100:.1f}%")
    
    results = {
        'removal_results': removal_results,
        'pairwise_subspace_cos': pairwise_cos,
        'n_subspaces': len(all_bases),
        'total_remaining_after_all': float(np.mean(np.sum(W_U_current**2, axis=1)) / original_energy),
    }
    
    print("\n=== P738 完成 ===")
    return results


# ============================================================
# P739: 完整子空间分解 -> 排序公式
# ============================================================

def P739_full_decomposition_ranking(model, tokenizer, device, model_info, model_name):
    """
    P739: 完整子空间分解 -> 排序公式
    - 用所有子空间重建logit
    - 目标: Top-1准确率 > 50%
    - 这将是编码机制的完整数学公式!
    """
    print("\n=== P739: 完整子空间分解 -> 排序公式 ===")
    
    W_U = to_numpy(get_W_U(model))
    n_vocab, d_model = W_U.shape
    n_layers = model_info.n_layers
    
    # 构建所有子空间基
    all_cats = {
        'syntax': SYNTAX_CATS,
        'semantic': SEMANTIC_CATS,
        'style': STYLE_CATS,
        'tense': TENSE_CATS,
        'number': NUMBER_CATS,
        'polarity': POLARITY_CATS,
        'topic': TOPIC_CATS,
    }
    
    all_bases = {}
    for dim_name, cats in all_cats.items():
        ids = get_word_ids(cats, tokenizer)
        basis = build_subspace_basis(ids, W_U)
        if basis is not None:
            all_bases[dim_name] = basis
    
    # 测试句子
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
    
    layers = get_layers(model)
    
    top1_correct = 0
    top5_correct = 0
    top10_correct = 0
    total_sentences = 0
    
    for sent_idx, sent in enumerate(TEST_SENTENCES):
        print(f"\n  分析句子: '{sent}'")
        
        tokens = tokenizer.encode(sent, return_tensors='pt').to(device)
        
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
        
        if not layer_outputs:
            continue
        
        last_layer_idx = max(layer_outputs.keys())
        last_pos = tokens.shape[1] - 1
        
        if len(layer_outputs[last_layer_idx].shape) == 3:
            h_final = to_numpy(layer_outputs[last_layer_idx][0, last_pos])
        else:
            continue
        
        # 原始logits
        logits_full = h_final @ W_U.T  # [n_vocab]
        top10_full = np.argsort(logits_full)[-10:][::-1]
        top1_word = tokenizer.decode([top10_full[0]]).strip().encode('ascii', 'replace').decode()
        
        # 用各子空间重建logits
        logits_reconstructed = np.zeros(n_vocab)
        word_contributions = {}  # 对top5词的各子空间贡献
        
        for dim_name, basis in all_bases.items():
            # h在子空间的投影
            h_proj = basis.T @ (basis @ h_final)
            # 该子空间对logits的贡献
            contrib = h_proj @ W_U.T  # [n_vocab]
            logits_reconstructed += contrib
            
            # top5词的贡献
            for rank in range(min(5, len(top10_full))):
                word_idx = top10_full[rank]
                word_name = tokenizer.decode([word_idx]).strip().encode('ascii', 'replace').decode()
                key = f"rank{rank}_{word_name}"
                if key not in word_contributions:
                    word_contributions[key] = {}
                word_contributions[key][dim_name] = float(contrib[word_idx])
        
        # 残差贡献
        h_in_all = np.zeros_like(h_final)
        for dim_name, basis in all_bases.items():
            h_in_all += basis.T @ (basis @ h_final)
        h_residual = h_final - h_in_all
        residual_contrib = h_residual @ W_U.T  # [n_vocab]
        logits_reconstructed += residual_contrib
        
        for rank in range(min(5, len(top10_full))):
            word_idx = top10_full[rank]
            word_name = tokenizer.decode([word_idx]).strip().encode('ascii', 'replace').decode()
            key = f"rank{rank}_{word_name}"
            if key not in word_contributions:
                word_contributions[key] = {}
            word_contributions[key]['residual'] = float(residual_contrib[word_idx])
        
        # 排序准确性
        top10_recon = np.argsort(logits_reconstructed)[-10:][::-1]
        top1_recon_word = tokenizer.decode([top10_recon[0]]).strip().encode('ascii', 'replace').decode()
        
        top1_match = top10_recon[0] == top10_full[0]
        top5_match = len(set(top10_recon[:5]) & set(top10_full[:5]))
        top10_match = len(set(top10_recon[:10]) & set(top10_full[:10]))
        
        if top1_match:
            top1_correct += 1
        top5_correct += top5_match
        top10_correct += top10_match
        total_sentences += 1
        
        # 不含残差的重建
        logits_no_residual = logits_reconstructed - residual_contrib
        top10_no_res = np.argsort(logits_no_residual)[-10:][::-1]
        top1_no_res_match = top10_no_res[0] == top10_full[0]
        
        # 各子空间贡献的logits的余弦相似度
        cos_full = np.dot(logits_reconstructed, logits_full) / (
            np.linalg.norm(logits_reconstructed) * np.linalg.norm(logits_full) + 1e-10)
        cos_no_res = np.dot(logits_no_residual, logits_full) / (
            np.linalg.norm(logits_no_residual) * np.linalg.norm(logits_full) + 1e-10)
        
        results[f'sentence_{sent_idx}'] = {
            'text': sent,
            'top1_full': top1_word,
            'top1_recon': top1_recon_word,
            'top1_match': bool(top1_match),
            'top5_overlap': int(top5_match),
            'top10_overlap': int(top10_match),
            'cosine_with_full': float(cos_full),
            'cosine_no_residual': float(cos_no_res),
            'top1_no_residual_match': bool(top1_no_res_match),
            'word_contributions': word_contributions,
        }
        
        print(f"    原始Top1: {top1_word}, 重建Top1: {top1_recon_word}, 匹配: {top1_match}")
        print(f"    Top5重叠: {top5_match}/5, Top10重叠: {top10_match}/10")
        print(f"    Cos(重建,原始)={cos_full:.4f}, Cos(无残差,原始)={cos_no_res:.4f}")
        
        del layer_outputs
        gc.collect()
        torch.cuda.empty_cache()
    
    # 汇总
    if total_sentences > 0:
        results['summary'] = {
            'top1_accuracy': float(top1_correct / total_sentences),
            'avg_top5_overlap': float(top5_correct / (total_sentences * 5)),
            'avg_top10_overlap': float(top10_correct / (total_sentences * 10)),
            'n_subspaces': len(all_bases),
        }
        
        print(f"\n  汇总:")
        print(f"    Top-1准确率: {top1_correct}/{total_sentences} = {top1_correct/total_sentences*100:.0f}%")
        print(f"    Top-5平均重叠: {top5_correct/(total_sentences*5)*100:.0f}%")
        print(f"    Top-10平均重叠: {top10_correct/(total_sentences*10)*100:.0f}%")
    
    results['conclusion'] = '子空间分解可有效重建排序' if top1_correct/total_sentences > 0.3 else '子空间分解不足以重建排序,残差空间包含关键信息'
    
    print("\n=== P739 完成 ===")
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
    print(f"Phase CLXXII: 残差空间分解 -- {model_name}")
    print(f"{'='*60}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    
    results = {}
    
    # P737
    try:
        results["P737"] = P737_residual_pca(model, tokenizer, device, model_info, model_name)
    except Exception as e:
        print(f"P737失败: {e}")
        import traceback
        traceback.print_exc()
        results["P737"] = {"error": str(e)}
    
    # P738
    try:
        results["P738"] = P738_residual_semantic(model, tokenizer, device, model_info, model_name)
    except Exception as e:
        print(f"P738失败: {e}")
        import traceback
        traceback.print_exc()
        results["P738"] = {"error": str(e)}
    
    # P739
    try:
        results["P739"] = P739_full_decomposition_ranking(model, tokenizer, device, model_info, model_name)
    except Exception as e:
        print(f"P739失败: {e}")
        import traceback
        traceback.print_exc()
        results["P739"] = {"error": str(e)}
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    output_dir = Path(f"results/phase_clxxii")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_name}_results.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n结果已保存到: {output_file}")


if __name__ == '__main__':
    main()
