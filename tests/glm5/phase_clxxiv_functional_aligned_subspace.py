"""
Phase CLXXIV: 功能对齐子空间发现
=================================
核心目标: 找到既保持W_U低秩结构, 又与语法/语义/风格对齐的子空间分解

上一阶段(CLXXIII)发现:
  - SVD按能量排序: 最大分量=最粗粒度区分(语言类型)
  - 语法/语义/风格是更精细的功能划分, 在SVD中分散在多个分量中
  - DS7B仅5个SVD分量即可100%重建logit, Qwen3需要500+仍不够

本阶段实验:
  P743: SVD子空间 × 功能投影
    - 将7类词(syntax/semantic/style/tense/number/polarity/topic)投影到SVD空间
    - 分析每类词在SVD子空间中的能量分布
    - 假说: 语法词集中在某些SVD分量, 语义词在另一些 → 可按功能对SVD重排序

  P744: 稀疏子空间发现 (Sparse PCA + NMF)
    - 对W_U在SVD低秩近似后的残差做Sparse PCA
    - NMF分解: 非负约束可能更接近"功能子空间"
    - 假说: 稀疏分量对应更清晰的功能语义

  P745: 功能对齐分层重建
    - 按功能层次重建: 语言类型→语法→语义→风格→残余
    - 对比SVD随机排序 vs 功能排序的重建效率
    - 目标: 用更少的功能对齐分量达到更高的logit重建精度
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import gc
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from sklearn.decomposition import PCA, SparsePCA, NMF, FastICA

from model_utils import load_model, get_model_info, get_layers


def to_numpy(tensor_or_array):
    if isinstance(tensor_or_array, np.ndarray):
        return tensor_or_array.astype(np.float32)
    return tensor_or_array.detach().cpu().float().numpy().astype(np.float32)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ============================================================
# 词分类定义 (与Phase CLXXII一致)
# ============================================================

SYNTAX_CATS = {
    'noun': ['cat', 'dog', 'house', 'car', 'tree', 'book', 'water', 'food',
             'hand', 'head', 'eye', 'door', 'table', 'chair', 'window'],
    'verb': ['run', 'walk', 'eat', 'drink', 'see', 'hear', 'think', 'know',
             'make', 'take', 'give', 'come', 'go', 'sit', 'stand'],
    'adj': ['big', 'small', 'good', 'bad', 'hot', 'cold', 'new', 'old',
            'fast', 'slow', 'high', 'low', 'long', 'short', 'strong'],
    'adv': ['quickly', 'slowly', 'carefully', 'easily', 'always', 'never',
            'often', 'rarely', 'very', 'quite', 'really', 'almost', 'just', 'still'],
    'prep': ['in', 'on', 'at', 'to', 'from', 'with', 'by', 'for', 'of', 'about',
             'between', 'through', 'under', 'over', 'after', 'before'],
    'pron': ['he', 'she', 'it', 'they', 'we', 'you', 'me', 'him', 'her',
             'them', 'us', 'this', 'that', 'my', 'your', 'his'],
    'det': ['the', 'a', 'an', 'this', 'that', 'some', 'any', 'all', 'every',
            'each', 'no', 'both', 'few', 'many', 'much'],
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
                 'warm', 'kind', 'gentle', 'strong', 'fast', 'smart'],
    'negative': ['bad', 'terrible', 'sad', 'hate', 'ugly', 'dark',
                 'cold', 'cruel', 'harsh', 'weak', 'slow', 'stupid'],
}

TOPIC_CATS = {
    'nature': ['tree', 'river', 'mountain', 'ocean', 'forest', 'flower',
               'rain', 'snow', 'wind', 'sun', 'moon', 'star'],
    'technology': ['computer', 'phone', 'internet', 'software', 'data',
                   'network', 'digital', 'code', 'program', 'system'],
    'society': ['government', 'law', 'politics', 'economy', 'culture',
                'education', 'health', 'science', 'religion', 'family'],
    'person': ['man', 'woman', 'child', 'friend', 'teacher', 'doctor',
               'leader', 'worker', 'artist', 'writer', 'soldier'],
}

ALL_CATS = {
    'syntax': SYNTAX_CATS,
    'semantic': SEMANTIC_CATS,
    'style': STYLE_CATS,
    'tense': TENSE_CATS,
    'number': NUMBER_CATS,
    'polarity': POLARITY_CATS,
    'topic': TOPIC_CATS,
}


def get_word_ids(cats, tokenizer):
    """获取词的token ID (仅保留单token词)"""
    word_ids = {}
    for cat, words in cats.items():
        cat_ids = {}
        for w in words:
            tokens = tokenizer.encode(w, add_special_tokens=False)
            if len(tokens) == 1:
                cat_ids[w] = tokens[0]
        word_ids[cat] = cat_ids
    return word_ids


def compute_svd_gpu(W_U_gpu, n_vocab, d_model):
    """GPU加速的SVD分解 (通过W_U^T @ W_U特征分解)"""
    print("  Computing SVD via eigendecomposition of W_U^T @ W_U ...", flush=True)
    chunk_size = 50000
    n_chunks = (n_vocab + chunk_size - 1) // chunk_size
    WtW = torch.zeros(d_model, d_model, dtype=torch.float32, device='cuda')
    for ci in range(n_chunks):
        start = ci * chunk_size
        end = min((ci + 1) * chunk_size, n_vocab)
        chunk = W_U_gpu[start:end].float()
        WtW += chunk.T @ chunk
        print(f"    Chunk {ci+1}/{n_chunks} done", flush=True)

    print("  Eigendecomposition on GPU ...", flush=True)
    eigenvalues_t, eigenvectors_t = torch.linalg.eigh(WtW)
    idx = torch.argsort(eigenvalues_t, descending=True)
    eigenvalues_t = eigenvalues_t[idx]
    eigenvectors_t = eigenvectors_t[:, idx]
    S = torch.sqrt(torch.maximum(eigenvalues_t, torch.zeros_like(eigenvalues_t))).cpu().numpy()
    Vt = eigenvectors_t.T.cpu().numpy()
    del WtW, eigenvalues_t, eigenvectors_t
    torch.cuda.empty_cache()
    return S, Vt


# ============================================================
# P743: SVD子空间 × 功能投影
# ============================================================

def P743_svd_functional_projection(W_U_gpu, tokenizer, model_name, results):
    """
    分析7类功能词在SVD子空间中的能量分布

    核心思路:
    - 每个词w_i在W_U中的行向量 W_U[i] 可投影到SVD子空间
    - 投影系数 = W_U[i] · Vt[k] / S[k] (在k方向的分量)
    - 按功能类别聚合, 看各类词集中在哪些SVD分量
    """
    print("\n--- P743: SVD子空间 × 功能投影 ---")

    n_vocab, d_model = W_U_gpu.shape
    print(f"  W_U shape: {n_vocab} x {d_model}")

    # Step 1: 计算SVD
    S, Vt = compute_svd_gpu(W_U_gpu, n_vocab, d_model)
    total_energy = np.sum(S**2)
    print(f"  Top 10 SVs: {S[:10].tolist()}")

    # Step 2: 获取各类功能词的token ID
    all_word_ids = {}
    for dim_name, cats in ALL_CATS.items():
        ids = get_word_ids(cats, tokenizer)
        all_word_ids[dim_name] = ids

    # Step 3: 计算 W_U @ Vt (投影到SVD右奇异向量)
    # 每个词在SVD空间中的坐标 = W_U[i] @ Vt.T = (S[i] * U[i])
    # 直接用投影系数 proj[i,k] = W_U[i] · Vt[k]
    n_svd = min(200, d_model)  # 分析前200个SVD分量
    print(f"  Computing W_U @ Vt[:{n_svd}] on GPU ...", flush=True)
    Vt_torch = torch.tensor(Vt[:n_svd], dtype=torch.float32, device='cuda')
    chunk_size = 50000
    n_chunks = (n_vocab + chunk_size - 1) // chunk_size
    proj_chunks = []
    for ci in range(n_chunks):
        start = ci * chunk_size
        end = min((ci + 1) * chunk_size, n_vocab)
        chunk = W_U_gpu[start:end].float()
        proj_chunk = chunk @ Vt_torch.T  # [chunk, n_svd]
        proj_chunks.append(proj_chunk.cpu().numpy())
    proj_all = np.concatenate(proj_chunks, axis=0)  # [n_vocab, n_svd]
    print(f"  Projection matrix shape: {proj_all.shape}")

    # Step 4: 分析各类功能词在SVD空间中的能量分布
    print("\n  功能词在SVD空间中的能量分布:")
    functional_energy = {}

    for dim_name, dim_ids in all_word_ids.items():
        # 收集该维度下所有词的索引
        all_indices = []
        for cat, cat_ids in dim_ids.items():
            for w, idx in cat_ids.items():
                if idx < n_vocab:
                    all_indices.append((cat, w, idx))

        if not all_indices:
            continue

        # 计算每个词在SVD空间中的能量分布
        cat_energy = {}
        for cat, w, idx in all_indices:
            # 词在SVD空间中的投影能量
            word_proj = proj_all[idx]  # [n_svd]
            word_energy = word_proj**2
            total_word_energy = np.sum(word_energy) + 1e-30

            # 在前10/20/50/100/200个SVD分量的能量
            energy_top10 = float(np.sum(word_energy[:10]) / total_word_energy)
            energy_top50 = float(np.sum(word_energy[:50]) / total_word_energy)
            energy_top100 = float(np.sum(word_energy[:100]) / total_word_energy)
            energy_top200 = float(np.sum(word_energy[:200]) / total_word_energy)

            cat_energy[f"{cat}_{w}"] = {
                'energy_top10': energy_top10,
                'energy_top50': energy_top50,
                'energy_top100': energy_top100,
                'energy_top200': energy_top200,
                'dominant_sv': int(np.argmax(word_energy)),
                'max_sv_energy': float(np.max(word_energy) / total_word_energy),
            }

        # 按类别汇总
        dim_summary = {}
        for cat in dim_ids:
            cat_words = {k: v for k, v in cat_energy.items() if k.startswith(f"{cat}_")}
            if not cat_words:
                continue
            avg_top10 = np.mean([v['energy_top10'] for v in cat_words.values()])
            avg_top50 = np.mean([v['energy_top50'] for v in cat_words.values()])
            avg_top100 = np.mean([v['energy_top100'] for v in cat_words.values()])
            avg_dominant = np.mean([v['dominant_sv'] for v in cat_words.values()])
            dim_summary[cat] = {
                'avg_energy_top10': float(avg_top10),
                'avg_energy_top50': float(avg_top50),
                'avg_energy_top100': float(avg_top100),
                'avg_dominant_sv': float(avg_dominant),
            }

        functional_energy[dim_name] = dim_summary

        # 打印汇总
        print(f"\n  {dim_name}:")
        for cat, stats in dim_summary.items():
            print(f"    {cat}: top10={stats['avg_energy_top10']*100:.1f}%, "
                  f"top50={stats['avg_energy_top50']*100:.1f}%, "
                  f"avg_dominant_sv={stats['avg_dominant_sv']:.0f}")

    # Step 5: SVD分量的"功能指纹"
    # 对每个SVD分量, 计算各类功能词在该分量上的平均投影能量
    # 注意: 计算所有n_svd个分量的指纹(后续功能分配需要)
    print("\n  SVD分量的功能指纹 (前20个分量):")
    sv_fingerprint = {}
    for k in range(n_svd):
        fingerprint = {}
        for dim_name, dim_ids in all_word_ids.items():
            all_indices = []
            for cat, cat_ids in dim_ids.items():
                for w, idx in cat_ids.items():
                    if idx < n_vocab:
                        all_indices.append(idx)
            if all_indices:
                # 该功能类词在第k个SVD分量的平均投影
                func_proj = proj_all[all_indices, k]
                func_energy = np.mean(func_proj**2)
                # 全词表的平均投影能量作为基线
                baseline_energy = np.mean(proj_all[:, k]**2)
                enrichment = func_energy / (baseline_energy + 1e-30)
                fingerprint[dim_name] = {
                    'energy': float(func_energy),
                    'baseline': float(baseline_energy),
                    'enrichment': float(enrichment),
                }
        sv_fingerprint[f"sv_{k}"] = fingerprint

        # 只打印前20个分量的指纹
        if k < 20:
            enriched = sorted(fingerprint.items(), key=lambda x: x[1]['enrichment'], reverse=True)
            top_enriched = enriched[:3]
            enrich_str = ", ".join([f"{n}={v['enrichment']:.2f}x" for n, v in top_enriched])
            print(f"    SV{k} (σ={S[k]:.2f}): {enrich_str}")

    # Step 6: 功能对齐的SVD重排序
    # 对每个功能维度, 找到该维度最富集的SVD分量集合
    print("\n  功能对齐的SVD分量分配:")
    functional_sv_assignment = {}
    for dim_name in ALL_CATS:
        # 找到enrichment最高的SVD分量
        enrichments = []
        for k in range(n_svd):
            if dim_name in sv_fingerprint[f"sv_{k}"]:
                enrichments.append((k, sv_fingerprint[f"sv_{k}"][dim_name]['enrichment']))
        enrichments.sort(key=lambda x: x[1], reverse=True)
        top5_svs = enrichments[:5]
        functional_sv_assignment[dim_name] = {
            'top5_svs': [(int(k), float(e)) for k, e in top5_svs],
            'total_enrichment': float(sum(e for _, e in top5_svs)),
        }
        sv_str = ", ".join([f"SV{k}({e:.2f}x)" for k, e in top5_svs])
        print(f"    {dim_name}: {sv_str}")

    results["p743_svd_functional"] = {
        "functional_energy": functional_energy,
        "sv_fingerprint": sv_fingerprint,
        "functional_sv_assignment": functional_sv_assignment,
        "n_svd_analyzed": n_svd,
    }

    return S, Vt, proj_all, results


# ============================================================
# P744: 稀疏子空间发现
# ============================================================

def P744_sparse_subspace(W_U_gpu, tokenizer, model_name, results, S, Vt, proj_all):
    """
    Sparse PCA + NMF + ICA: 寻找功能对齐的稀疏子空间

    核心思路:
    - SVD给出密集的旋转基 → 难以解释
    - Sparse PCA给出稀疏基 → 每个分量只激活少量维度
    - NMF给出非负基 → 可能更接近"功能模块"
    - ICA给出独立基 → 可能对应独立的功能源
    """
    print("\n--- P744: 稀疏子空间发现 ---")

    n_vocab, d_model = W_U_gpu.shape

    # 使用SVD低秩近似来减少计算量
    # W_U ≈ U_k S_k V_k^T, 我们在V_k空间中工作
    n_components = min(100, d_model)
    print(f"  Working in top-{n_components} SVD subspace")

    # 将W_U投影到SVD子空间: W_U_proj = W_U @ Vt[:n_components].T  → [n_vocab, n_components]
    # 这已经在proj_all中计算了 (前n_components列)
    W_U_svd = proj_all[:, :n_components]  # [n_vocab, n_components]

    # 获取功能词索引
    all_word_ids = {}
    for dim_name, cats in ALL_CATS.items():
        ids = get_word_ids(cats, tokenizer)
        all_word_ids[dim_name] = ids

    # ---- Sparse PCA ----
    print("\n  Sparse PCA (n_components=30, alpha=0.5) ...", flush=True)
    try:
        spca = SparsePCA(n_components=30, alpha=0.5, random_state=42, max_iter=100)
        # 只用一部分词来加速 (采样+功能词)
        n_sample = min(20000, n_vocab)
        sample_idx = np.random.choice(n_vocab, n_sample, replace=False)
        # 确保功能词在样本中
        func_indices = set()
        for dim_name, dim_ids in all_word_ids.items():
            for cat, cat_ids in dim_ids.items():
                for w, idx in cat_ids.items():
                    if idx < n_vocab:
                        func_indices.add(idx)
        func_indices = list(func_indices)
        sample_idx = np.unique(np.concatenate([sample_idx, func_indices]))
        np.random.shuffle(sample_idx)

        W_U_sample = W_U_svd[sample_idx]
        spca_components = spca.fit_transform(W_U_sample.T).T  # [30, n_components]
        print(f"  Sparse PCA done. Components shape: {spca_components.shape}")

        # 分析Sparse PCA分量的功能指纹
        spca_fingerprint = {}
        for k in range(min(10, spca_components.shape[0])):
            # 每个SPCA分量在原始SVD空间的方向
            comp = spca_components[k]
            fingerprint = {}
            for dim_name, dim_ids in all_word_ids.items():
                all_indices = []
                for cat, cat_ids in dim_ids.items():
                    for w, idx in cat_ids.items():
                        if idx < n_vocab:
                            all_indices.append(idx)
                if all_indices:
                    # 功能词在SPCA分量k上的投影
                    func_proj = W_U_svd[all_indices] @ comp  # [n_func]
                    func_energy = np.mean(func_proj**2)
                    baseline_energy = np.mean((W_U_svd[sample_idx] @ comp)**2)
                    enrichment = func_energy / (baseline_energy + 1e-30)
                    fingerprint[dim_name] = float(enrichment)

            spca_fingerprint[f"spca_{k}"] = fingerprint
            enriched = sorted(fingerprint.items(), key=lambda x: x[1], reverse=True)
            top3 = enriched[:3]
            enrich_str = ", ".join([f"{n}={v:.2f}x" for n, v in top3])
            print(f"    SPCA{k}: {enrich_str}")

    except Exception as e:
        print(f"  Sparse PCA failed: {e}")
        spca_fingerprint = {"error": str(e)}

    # ---- NMF ----
    print("\n  NMF (n_components=30) ...", flush=True)
    try:
        # NMF需要非负输入, 对W_U_svd做shift
        W_U_nmf_input = W_U_svd[sample_idx] - W_U_svd[sample_idx].min() + 1e-6
        nmf = NMF(n_components=30, random_state=42, max_iter=200)
        W_nmf = nmf.fit_transform(W_U_nmf_input)  # [n_sample, 30]
        H_nmf = nmf.components_  # [30, n_components]
        print(f"  NMF done. H shape: {H_nmf.shape}")

        # 分析NMF分量的功能指纹
        nmf_fingerprint = {}
        for k in range(min(10, H_nmf.shape[0])):
            comp = H_nmf[k]
            fingerprint = {}
            for dim_name, dim_ids in all_word_ids.items():
                all_indices = []
                for cat, cat_ids in dim_ids.items():
                    for w, idx in cat_ids.items():
                        if idx < n_vocab:
                            all_indices.append(idx)
                if all_indices:
                    func_proj = W_U_svd[all_indices] @ comp
                    func_energy = np.mean(func_proj**2)
                    baseline_energy = np.mean((W_U_svd[sample_idx] @ comp)**2)
                    enrichment = func_energy / (baseline_energy + 1e-30)
                    fingerprint[dim_name] = float(enrichment)

            nmf_fingerprint[f"nmf_{k}"] = fingerprint
            enriched = sorted(fingerprint.items(), key=lambda x: x[1], reverse=True)
            top3 = enriched[:3]
            enrich_str = ", ".join([f"{n}={v:.2f}x" for n, v in top3])
            print(f"    NMF{k}: {enrich_str}")

        # NMF重建误差
        nmf_reconstruction = W_nmf @ H_nmf
        nmf_error = np.mean((W_U_nmf_input - nmf_reconstruction)**2) / np.mean(W_U_nmf_input**2)
        print(f"  NMF reconstruction error: {nmf_error*100:.1f}%")

    except Exception as e:
        print(f"  NMF failed: {e}")
        nmf_fingerprint = {"error": str(e)}
        nmf_error = -1

    # ---- ICA ----
    print("\n  FastICA (n_components=30) ...", flush=True)
    try:
        ica = FastICA(n_components=30, random_state=42, max_iter=200)
        # ICA在样本上拟合: 输入 [n_sample, n_components]
        S_ica = ica.fit_transform(W_U_svd[sample_idx])  # [n_sample, 30]
        A_ica = ica.mixing_  # [n_components, 30]
        print(f"  ICA done. Mixing shape: {A_ica.shape}, Sources shape: {S_ica.shape}")

        # 分析ICA分量的功能指纹
        # ICA的独立分量在SVD空间中的方向 = mixing_的列
        ica_fingerprint = {}
        for k in range(min(10, A_ica.shape[1])):
            comp = A_ica[:, k]  # [n_components] — 在SVD子空间中的方向
            fingerprint = {}
            for dim_name, dim_ids in all_word_ids.items():
                all_indices = []
                for cat, cat_ids in dim_ids.items():
                    for w, idx in cat_ids.items():
                        if idx < n_vocab:
                            all_indices.append(idx)
                if all_indices:
                    func_proj = W_U_svd[all_indices] @ comp  # [n_func]
                    func_energy = np.mean(func_proj**2)
                    baseline_energy = np.mean((W_U_svd[sample_idx] @ comp)**2)
                    enrichment = func_energy / (baseline_energy + 1e-30)
                    fingerprint[dim_name] = float(enrichment)

            ica_fingerprint[f"ica_{k}"] = fingerprint
            enriched = sorted(fingerprint.items(), key=lambda x: x[1], reverse=True)
            top3 = enriched[:3]
            enrich_str = ", ".join([f"{n}={v:.2f}x" for n, v in top3])
            print(f"    ICA{k}: {enrich_str}")

    except Exception as e:
        print(f"  ICA failed: {e}")
        ica_fingerprint = {"error": str(e)}

    # ---- 对比: SVD vs Sparse PCA vs NMF vs ICA 的功能对齐度 ----
    print("\n  方法对比 - 功能富集度 (各类维度最大富集的平均值):")
    method_enrichments = {}

    # SVD - 使用proj_all的列数作为n_svd
    n_svd_local = proj_all.shape[1]
    svd_enrichments = []
    for dim_name in ALL_CATS:
        best_enrich = 0
        for k in range(min(20, n_svd_local)):
            key = f"sv_{k}"
            if key in results.get("p743_svd_functional", {}).get("sv_fingerprint", {}):
                if dim_name in results["p743_svd_functional"]["sv_fingerprint"][key]:
                    e = results["p743_svd_functional"]["sv_fingerprint"][key][dim_name]["enrichment"]
                    best_enrich = max(best_enrich, e)
        svd_enrichments.append(best_enrich)
    method_enrichments['SVD'] = float(np.mean(svd_enrichments)) if svd_enrichments else 0

    # Sparse PCA
    if not isinstance(spca_fingerprint, dict) or "error" not in spca_fingerprint:
        spca_enrichments = []
        for dim_name in ALL_CATS:
            best_enrich = 0
            for k in range(min(10, 30)):
                key = f"spca_{k}"
                if key in spca_fingerprint and dim_name in spca_fingerprint[key]:
                    best_enrich = max(best_enrich, spca_fingerprint[key][dim_name])
            spca_enrichments.append(best_enrich)
        method_enrichments['SparsePCA'] = float(np.mean(spca_enrichments)) if spca_enrichments else 0

    # NMF
    if not isinstance(nmf_fingerprint, dict) or "error" not in nmf_fingerprint:
        nmf_enrichments = []
        for dim_name in ALL_CATS:
            best_enrich = 0
            for k in range(min(10, 30)):
                key = f"nmf_{k}"
                if key in nmf_fingerprint and dim_name in nmf_fingerprint[key]:
                    best_enrich = max(best_enrich, nmf_fingerprint[key][dim_name])
            nmf_enrichments.append(best_enrich)
        method_enrichments['NMF'] = float(np.mean(nmf_enrichments)) if nmf_enrichments else 0

    # ICA
    if not isinstance(ica_fingerprint, dict) or "error" not in ica_fingerprint:
        ica_enrichments = []
        for dim_name in ALL_CATS:
            best_enrich = 0
            for k in range(min(10, 30)):
                key = f"ica_{k}"
                if key in ica_fingerprint and dim_name in ica_fingerprint[key]:
                    best_enrich = max(best_enrich, ica_fingerprint[key][dim_name])
            ica_enrichments.append(best_enrich)
        method_enrichments['ICA'] = float(np.mean(ica_enrichments)) if ica_enrichments else 0

    for method, avg_enrich in method_enrichments.items():
        print(f"    {method}: {avg_enrich:.2f}x")

    results["p744_sparse_subspace"] = {
        "spca_fingerprint": spca_fingerprint,
        "nmf_fingerprint": nmf_fingerprint,
        "nmf_error": float(nmf_error) if nmf_error >= 0 else -1,
        "ica_fingerprint": ica_fingerprint,
        "method_enrichments": method_enrichments,
        "n_components": n_components,
        "n_sample": len(sample_idx) if 'sample_idx' in dir() else 0,
    }

    return results


# ============================================================
# P745: 功能对齐分层重建
# ============================================================

def P745_functional_hierarchical_reconstruction(model, tokenizer, device, model_name, results, S, Vt):
    """
    功能对齐的分层logit重建

    对比三种重建策略:
    1. SVD随机排序 (标准top-k)
    2. 功能排序: 按功能富集度对SVD分量重排
    3. 功能分层: 语言→语法→语义→风格→残余
    """
    print("\n--- P745: 功能对齐分层重建 ---")

    n_vocab, d_model = model.lm_head.weight.shape
    W_U_gpu = model.lm_head.weight.data.float()

    # 测试句子 (英文+中文混合)
    sentences = [
        "The cat sat on the",
        "I went to the store to buy",
        "She looked at him with a",
        "The weather was very cold and",
        "He opened the door and saw",
        "In the morning she always drinks",
        "The children played in the park",
        "After dinner they went for a",
        "The book was about a young",
        "They decided to go to the",
        "今天天气非常",           # Chinese
        "他去学校的时候",        # Chinese
        "这个问题的答案是",       # Chinese
    ]

    # 获取功能对齐的SVD分量分配
    func_assignment = results.get("p743_svd_functional", {}).get("functional_sv_assignment", {})

    # 为每个功能维度分配SVD分量
    # 格式: {dim_name: [sv_index, ...]}
    functional_sv_sets = {}
    for dim_name, info in func_assignment.items():
        functional_sv_sets[dim_name] = [k for k, _ in info.get('top5_svs', [])]

    # 注册hook
    layer_outputs = {}
    def hook_fn(module, input, output):
        layer_outputs["last_hidden"] = input[0]

    layers = model.model.layers
    last_layer = layers[-1]
    handle = last_layer.register_forward_hook(hook_fn)

    # 三种重建策略的k值
    k_values = [5, 10, 20, 50, 100, 200]

    # 策略1: 标准SVD top-k
    accuracy_svd = {k: {"top1": 0, "top5": 0, "top10": 0, "cosine": 0.0} for k in k_values}

    # 策略2: 功能排序 - 按功能富集度降序排列SVD分量
    # 构建功能排序的SVD分量索引
    all_sv_enrichments = []
    for k in range(min(200, len(S))):
        key = f"sv_{k}"
        fp = results.get("p743_svd_functional", {}).get("sv_fingerprint", {}).get(key, {})
        max_enrich = max([v.get('enrichment', 0) for v in fp.values()]) if fp else 0
        all_sv_enrichments.append((k, max_enrich))
    all_sv_enrichments.sort(key=lambda x: x[1], reverse=True)
    functional_order = [k for k, _ in all_sv_enrichments]

    accuracy_func = {k: {"top1": 0, "top5": 0, "top10": 0, "cosine": 0.0} for k in k_values}

    # 策略3: 功能分层
    # 层次: 语言区分(SV0) → 语法 → 语义 → 风格 → 时态 → 数量 → 极性 → 主题 → 残余
    hierarchy_order = [0]  # SV0 总是第一个
    for dim_name in ['syntax', 'semantic', 'style', 'tense', 'number', 'polarity', 'topic']:
        if dim_name in functional_sv_sets:
            hierarchy_order.extend([k for k in functional_sv_sets[dim_name] if k not in hierarchy_order])
    # 添加剩余分量
    for k in range(len(S)):
        if k not in hierarchy_order:
            hierarchy_order.append(k)

    accuracy_hier = {k: {"top1": 0, "top5": 0, "top10": 0, "cosine": 0.0} for k in k_values}

    n_valid = 0
    Vt_np = Vt  # [d_model, d_model]

    for sent_idx, sentence in enumerate(sentences):
        try:
            inputs = tokenizer(sentence, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
        except Exception as e:
            print(f"  Sentence {sent_idx+1} failed: {e}")
            continue

        h_final = to_numpy(layer_outputs["last_hidden"][0, -1, :])

        # 原始logits (GPU)
        with torch.no_grad():
            h_torch = torch.tensor(h_final, dtype=torch.float32, device=device)
            original_logits_torch = h_torch @ W_U_gpu.T
            original_logits = original_logits_torch.cpu().numpy()

        top10_original = set(np.argsort(original_logits)[-10:][::-1])
        top5_original = set(np.argsort(original_logits)[-5:][::-1])
        top1_original = np.argmax(original_logits)

        # h在SVD右奇异向量上的投影
        h_proj_vt = h_final @ Vt_np.T  # [d_model]

        for k in k_values:
            # 策略1: 标准SVD top-k
            h_svd = Vt_np[:k].T @ h_proj_vt[:k]
            with torch.no_grad():
                h_t = torch.tensor(h_svd, dtype=torch.float32, device=device)
                recon_t = W_U_gpu @ h_t
                recon = recon_t.cpu().numpy()
            accuracy_svd[k]["top1"] += (1 if np.argmax(recon) == top1_original else 0)
            accuracy_svd[k]["top5"] += len(set(np.argsort(recon)[-5:][::-1]) & top5_original) / 5.0
            accuracy_svd[k]["top10"] += len(set(np.argsort(recon)[-10:][::-1]) & top10_original) / 10.0
            cos = np.dot(original_logits, recon) / (np.linalg.norm(original_logits) * np.linalg.norm(recon) + 1e-30)
            accuracy_svd[k]["cosine"] += cos

            # 策略2: 功能排序 top-k
            func_idx = functional_order[:k]
            h_func = Vt_np[func_idx].T @ h_proj_vt[func_idx]
            with torch.no_grad():
                h_t = torch.tensor(h_func, dtype=torch.float32, device=device)
                recon_t = W_U_gpu @ h_t
                recon = recon_t.cpu().numpy()
            accuracy_func[k]["top1"] += (1 if np.argmax(recon) == top1_original else 0)
            accuracy_func[k]["top5"] += len(set(np.argsort(recon)[-5:][::-1]) & top5_original) / 5.0
            accuracy_func[k]["top10"] += len(set(np.argsort(recon)[-10:][::-1]) & top10_original) / 10.0
            cos = np.dot(original_logits, recon) / (np.linalg.norm(original_logits) * np.linalg.norm(recon) + 1e-30)
            accuracy_func[k]["cosine"] += cos

            # 策略3: 功能分层 top-k
            hier_idx = hierarchy_order[:k]
            h_hier = Vt_np[hier_idx].T @ h_proj_vt[hier_idx]
            with torch.no_grad():
                h_t = torch.tensor(h_hier, dtype=torch.float32, device=device)
                recon_t = W_U_gpu @ h_t
                recon = recon_t.cpu().numpy()
            accuracy_hier[k]["top1"] += (1 if np.argmax(recon) == top1_original else 0)
            accuracy_hier[k]["top5"] += len(set(np.argsort(recon)[-5:][::-1]) & top5_original) / 5.0
            accuracy_hier[k]["top10"] += len(set(np.argsort(recon)[-10:][::-1]) & top10_original) / 10.0
            cos = np.dot(original_logits, recon) / (np.linalg.norm(original_logits) * np.linalg.norm(recon) + 1e-30)
            accuracy_hier[k]["cosine"] += cos

        n_valid += 1
        print(f"  Sentence {sent_idx+1}: done")

    handle.remove()

    # 平均
    for k in k_values:
        for metric in ["top1", "top5", "top10", "cosine"]:
            accuracy_svd[k][metric] /= max(n_valid, 1)
            accuracy_func[k][metric] /= max(n_valid, 1)
            accuracy_hier[k][metric] /= max(n_valid, 1)

    # 打印对比表
    print(f"\n  分层重建对比 (n_sentences={n_valid}):")
    print(f"  {'k':>5} | {'SVD-Top1':>9} | {'Func-Top1':>9} | {'Hier-Top1':>9} | "
          f"{'SVD-Cos':>8} | {'Func-Cos':>8} | {'Hier-Cos':>8}")
    print("  " + "-" * 75)
    for k in k_values:
        print(f"  {k:>5} | {accuracy_svd[k]['top1']:>9.2f} | {accuracy_func[k]['top1']:>9.2f} | "
              f"{accuracy_hier[k]['top1']:>9.2f} | "
              f"{accuracy_svd[k]['cosine']:>8.4f} | {accuracy_func[k]['cosine']:>8.4f} | "
              f"{accuracy_hier[k]['cosine']:>8.4f}")

    results["p745_hierarchical"] = {
        "accuracy_svd": {str(k): v for k, v in accuracy_svd.items()},
        "accuracy_func": {str(k): v for k, v in accuracy_func.items()},
        "accuracy_hier": {str(k): v for k, v in accuracy_hier.items()},
        "n_valid_sentences": n_valid,
        "functional_order_top20": functional_order[:20],
        "hierarchy_order_top20": hierarchy_order[:20],
    }

    return results


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "deepseek7b", "glm4"])
    args = parser.parse_args()

    model_name = args.model
    print(f"Phase CLXXIV: 功能对齐子空间发现 -- {model_name}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 加载模型
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)

    # 获取W_U (GPU)
    W_U_gpu = model.lm_head.weight.data  # [n_vocab, d_model] on GPU
    n_vocab, d_model = W_U_gpu.shape
    print(f"  W_U shape: {n_vocab} x {d_model}")

    results = {"model": model_name, "timestamp": datetime.now().isoformat()}

    # P743: SVD × 功能投影
    S, Vt, proj_all, results = P743_svd_functional_projection(W_U_gpu, tokenizer, model_name, results)

    # P744: 稀疏子空间发现
    results = P744_sparse_subspace(W_U_gpu, tokenizer, model_name, results, S, Vt, proj_all)

    # P745: 功能对齐分层重建
    results = P745_functional_hierarchical_reconstruction(model, tokenizer, device, model_name, results, S, Vt)

    # 释放模型
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # 保存
    out_dir = Path(f"results/phase_clxxiv")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{model_name}_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    print(f"\nResults saved to {out_file}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
