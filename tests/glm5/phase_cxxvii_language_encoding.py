"""
Phase CXXVII-CXXVIII: 语言编码的数学机制
P559: W_U空间中的语义方向 — 不同token在W_U空间中形成什么结构?
P560: 频谱如何编码语法vs语义 — ratio(k)与token类型/语法角色的关系
P561: MLP神经元的"功能专业化" — 不同神经元编码什么语义信息?
P562: 从频谱力学到语言能力的映射 — 频谱结构如何支持下一个token预测?
"""

import argparse
import json
import os
import sys
import numpy as np
from scipy.linalg import svd
from scipy.sparse.linalg import svds
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from model_utils import load_model, get_model_info, get_W_U, get_sample_layers, get_layers, get_layer_weights


def compute_wu_svd(model, k=200):
    """计算W_U的SVD"""
    W_U = get_W_U(model)
    d_model, n_vocab = W_U.shape
    k = min(k, min(d_model, n_vocab) - 1)
    U_wu, S_wu, Vt_wu = svds(W_U.T, k=k)
    sort_idx = np.argsort(S_wu)[::-1]
    U_wu = U_wu[:, sort_idx]
    S_wu = S_wu[sort_idx]
    return U_wu, S_wu, W_U


def project_and_spectrum(h, U_wu, k):
    """将h投影到W_U空间并计算频谱"""
    coeffs = U_wu[:, :k].T @ h
    spectrum = np.sort(np.abs(coeffs))[::-1]
    return spectrum


def get_hidden_states(model, inputs, n_layers):
    """获取所有层的hidden states"""
    with torch.no_grad():
        outputs = model(inputs["input_ids"], output_hidden_states=True)
    h_states = outputs.hidden_states
    return [h for h in h_states]


import torch


def compute_ratio_k(h, U_wu, k):
    """计算ratio(k) = ||proj||^2 / ||h||^2"""
    h_np = h.detach().cpu().float().numpy() if isinstance(h, torch.Tensor) else h
    proj = U_wu[:, :k].T @ h_np
    ratio = np.sum(proj**2) / (np.sum(h_np**2) + 1e-10)
    return ratio


# ============== P559: W_U空间中的语义方向 ==============
def experiment_p559(model, tokenizer, model_name, device, U_wu, S_wu, W_U):
    """W_U空间中的语义方向 — 不同token在W_U空间中形成什么结构?"""
    print(f"\n{'='*60}")
    print(f"P559: W_U空间中的语义方向 — {model_name}")
    print(f"{'='*60}")
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    k_wu = min(200, U_wu.shape[1])
    
    # 定义语义类别和代表token
    semantic_categories = {
        "动物": ["cat", "dog", "bird", "fish", "horse", "lion", "tiger", "bear", "wolf", "fox"],
        "颜色": ["red", "blue", "green", "yellow", "white", "black", "pink", "brown", "gray", "orange"],
        "数字": ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"],
        "动词": ["run", "walk", "eat", "sleep", "think", "speak", "write", "read", "play", "work"],
        "形容词": ["big", "small", "fast", "slow", "hot", "cold", "old", "new", "good", "bad"],
        "名词_抽象": ["love", "time", "life", "world", "mind", "power", "truth", "beauty", "freedom", "justice"],
        "名词_具体": ["house", "car", "book", "tree", "water", "fire", "stone", "door", "table", "chair"],
        "功能词": ["the", "a", "an", "is", "are", "was", "were", "have", "has", "do"],
    }
    
    # 1. 计算每个token在W_U空间中的坐标
    token_coords = {}  # token -> (category, coord in W_U space)
    category_coords = defaultdict(list)
    
    for cat, tokens in semantic_categories.items():
        for token in tokens:
            tok_ids = tokenizer.encode(token, add_special_tokens=False)
            if len(tok_ids) == 0:
                continue
            tok_id = tok_ids[0]
            # token在W_U空间中的坐标 = W_U的行(即lm_head的行)
            coord = W_U[tok_id]  # [d_model]
            # 投影到前k_wu个W_U奇异方向
            proj = U_wu[:, :k_wu].T @ coord
            token_coords[token] = (cat, proj)
            category_coords[cat].append(proj)
    
    print(f"\n有效token数: {len(token_coords)}")
    
    # 2. 类内相似度 vs 类间相似度
    print("\n--- 类内/类间余弦相似度 ---")
    intra_sims = []
    inter_sims = []
    
    for cat, coords in category_coords.items():
        if len(coords) < 2:
            continue
        # 类内
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                c1, c2 = coords[i], coords[j]
                norm1, norm2 = np.linalg.norm(c1), np.linalg.norm(c2)
                if norm1 > 0 and norm2 > 0:
                    sim = np.dot(c1, c2) / (norm1 * norm2)
                    intra_sims.append(sim)
        # 类间
        for other_cat, other_coords in category_coords.items():
            if other_cat == cat:
                continue
            for c1 in coords[:3]:
                for c2 in other_coords[:3]:
                    norm1, norm2 = np.linalg.norm(c1), np.linalg.norm(c2)
                    if norm1 > 0 and norm2 > 0:
                        sim = np.dot(c1, c2) / (norm1 * norm2)
                        inter_sims.append(sim)
    
    intra_mean = np.mean(intra_sims) if intra_sims else 0
    inter_mean = np.mean(inter_sims) if inter_sims else 0
    print(f"类内平均余弦相似度: {intra_mean:.4f}")
    print(f"类间平均余弦相似度: {inter_mean:.4f}")
    print(f"类内>类间? {'是' if intra_mean > inter_mean else '否'}")
    
    # 3. 每个类别的频谱分布
    print("\n--- 各类别的频谱特征 ---")
    category_spectra = {}
    for cat, coords in category_coords.items():
        if len(coords) == 0:
            continue
        # 类平均频谱
        mean_spec = np.mean(np.abs(np.array(coords)), axis=0)
        # ratio(10)和ratio(50)
        ratios_10 = []
        ratios_50 = []
        for coord in coords:
            abs_coord = np.abs(coord)
            total = np.sum(abs_coord**2) + 1e-10
            r10 = np.sum(abs_coord[:10]**2) / total
            r50 = np.sum(abs_coord[:50]**2) / total
            ratios_10.append(r10)
            ratios_50.append(r50)
        category_spectra[cat] = {
            "ratio_10_mean": np.mean(ratios_10),
            "ratio_50_mean": np.mean(ratios_50),
            "ratio_10_std": np.std(ratios_10),
            "ratio_50_std": np.std(ratios_50),
        }
        print(f"  {cat}: ratio(10)={np.mean(ratios_10):.3f}±{np.std(ratios_10):.3f}, "
              f"ratio(50)={np.mean(ratios_50):.3f}±{np.std(ratios_50):.3f}")
    
    # 4. 不同类别在W_U主方向上的分布
    print("\n--- 各类别在W_U主方向上的分布 ---")
    direction_concentration = {}
    for cat, coords in category_coords.items():
        if len(coords) == 0:
            continue
        # 每个coord在W_U各方向上的能量
        all_energies = np.abs(np.array(coords))  # [n_tokens, k_wu]
        mean_energies = np.mean(all_energies**2, axis=0)  # [k_wu]
        # 前3个方向的集中度
        top3_ratio = np.sum(mean_energies[:3]) / (np.sum(mean_energies) + 1e-10)
        top5_ratio = np.sum(mean_energies[:5]) / (np.sum(mean_energies) + 1e-10)
        top10_ratio = np.sum(mean_energies[:10]) / (np.sum(mean_energies) + 1e-10)
        direction_concentration[cat] = {
            "top3_ratio": top3_ratio,
            "top5_ratio": top5_ratio,
            "top10_ratio": top10_ratio,
        }
        print(f"  {cat}: Top-3集中={top3_ratio:.3f}, Top-5={top5_ratio:.3f}, Top-10={top10_ratio:.3f}")
    
    # 5. 功能词 vs 内容词的频谱差异
    print("\n--- 功能词 vs 内容词 ---")
    function_cats = ["功能词"]
    content_cats = [c for c in category_coords.keys() if c not in function_cats]
    
    func_ratio10 = []
    func_ratio50 = []
    content_ratio10 = []
    content_ratio50 = []
    
    for cat in function_cats:
        for coord in category_coords[cat]:
            abs_coord = np.abs(coord)
            total = np.sum(abs_coord**2) + 1e-10
            func_ratio10.append(np.sum(abs_coord[:10]**2) / total)
            func_ratio50.append(np.sum(abs_coord[:50]**2) / total)
    
    for cat in content_cats:
        for coord in category_coords[cat]:
            abs_coord = np.abs(coord)
            total = np.sum(abs_coord**2) + 1e-10
            content_ratio10.append(np.sum(abs_coord[:10]**2) / total)
            content_ratio50.append(np.sum(abs_coord[:50]**2) / total)
    
    print(f"功能词: ratio(10)={np.mean(func_ratio10):.3f}±{np.std(func_ratio10):.3f}, "
          f"ratio(50)={np.mean(func_ratio50):.3f}±{np.std(func_ratio50):.3f}")
    print(f"内容词: ratio(10)={np.mean(content_ratio10):.3f}±{np.std(content_ratio10):.3f}, "
          f"ratio(50)={np.mean(content_ratio50):.3f}±{np.std(content_ratio50):.3f}")
    
    # 6. 在W_U空间中做PCA看类别结构
    print("\n--- W_U空间中的类别聚类结构 ---")
    all_coords_list = []
    all_labels = []
    for cat, coords in category_coords.items():
        for coord in coords:
            all_coords_list.append(coord)
            all_labels.append(cat)
    
    if len(all_coords_list) > 5:
        X = np.array(all_coords_list)  # [n_tokens, k_wu]
        # 减均值
        X_centered = X - X.mean(axis=0, keepdims=True)
        # PCA
        cov = X_centered.T @ X_centered / len(X_centered)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = eigvals[::-1]
        eigvecs = eigvecs[:, ::-1]
        # 前3个主成分的方差解释率
        var_explained = eigvals / (np.sum(eigvals) + 1e-10)
        print(f"前3个主成分方差解释率: PC1={var_explained[0]:.3f}, PC2={var_explained[1]:.3f}, PC3={var_explained[2]:.3f}")
        print(f"前10个PC累积解释率: {np.sum(var_explained[:10]):.3f}")
        
        # 投影到前2个主成分
        proj_pc = X_centered @ eigvecs[:, :2]  # [n_tokens, 2]
        
        # 各类别在PC1和PC2上的中心
        print("各类别在PC1-PC2上的中心:")
        for cat in category_coords.keys():
            idx = [i for i, l in enumerate(all_labels) if l == cat]
            if idx:
                center = proj_pc[idx].mean(axis=0)
                print(f"  {cat}: PC1={center[0]:.3f}, PC2={center[1]:.3f}")
    
    result = {
        "model": model_name,
        "intra_sim": float(intra_mean),
        "inter_sim": float(inter_mean),
        "category_spectra": category_spectra,
        "direction_concentration": direction_concentration,
        "func_ratio10_mean": float(np.mean(func_ratio10)),
        "func_ratio50_mean": float(np.mean(func_ratio50)),
        "content_ratio10_mean": float(np.mean(content_ratio10)),
        "content_ratio50_mean": float(np.mean(content_ratio50)),
    }
    
    return result


# ============== P560: 频谱如何编码语法vs语义 ==============
def experiment_p560(model, tokenizer, model_name, device, U_wu, S_wu, W_U):
    """频谱如何编码语法vs语义 — ratio(k)与token类型/语法角色的关系"""
    print(f"\n{'='*60}")
    print(f"P560: 频谱编码语法vs语义 — {model_name}")
    print(f"{'='*60}")
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    
    # 1. 构造不同语法角色的句子
    sentences = [
        # 主语
        ("The cat sat on the mat", "cat", "subject"),
        ("A dog ran through the park", "dog", "subject"),
        ("The bird flew over the house", "bird", "subject"),
        # 宾语
        ("She saw the cat yesterday", "cat", "object"),
        ("He found a dog outside", "dog", "object"),
        ("They watched the bird carefully", "bird", "object"),
        # 修饰语
        ("The red car drove fast", "red", "modifier"),
        ("A big house stood there", "big", "modifier"),
        ("The old tree fell down", "old", "modifier"),
        # 功能词
        ("This is the best one", "the", "function"),
        ("He has a new book", "has", "function"),
        ("She was very happy", "was", "function"),
    ]
    
    # 2. 对每个句子，获取目标token在各层的ratio(k)
    k_values = [10, 30, 50, 100, 200]
    token_ratio_data = defaultdict(lambda: defaultdict(list))  # role -> k -> [ratios]
    
    for sent, target_word, role in sentences:
        inputs = tokenizer(sent, return_tensors="pt").to(device)
        tok_ids = tokenizer.encode(target_word, add_special_tokens=False)
        if not tok_ids:
            continue
        target_id = tok_ids[0]
        
        # 找到target token在句子中的位置
        sent_ids = inputs["input_ids"][0].cpu().tolist()
        positions = [i for i, tid in enumerate(sent_ids) if tid == target_id]
        if not positions:
            continue
        pos = positions[0]
        
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
        
        # 逐层计算ratio
        for layer_idx in [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]:
            h = outputs.hidden_states[layer_idx][0, pos].detach().cpu().float().numpy()
            for k in k_values:
                r = compute_ratio_k_from_h(h, U_wu, k)
                token_ratio_data[role][k].append(r)
    
    # 3. 对比不同语法角色的ratio(k)
    print("\n--- 不同语法角色的ratio(k) ---")
    for role in ["subject", "object", "modifier", "function"]:
        print(f"\n  {role}:")
        for k in k_values:
            vals = token_ratio_data[role][k]
            if vals:
                print(f"    ratio({k}): {np.mean(vals):.3f} ± {np.std(vals):.3f}")
    
    # 4. 逐层ratio(k)演化对比
    print("\n--- 逐层ratio(50)演化对比 ---")
    layer_ratio_data = defaultdict(lambda: defaultdict(list))  # role -> layer -> [ratios]
    
    for sent, target_word, role in sentences:
        inputs = tokenizer(sent, return_tensors="pt").to(device)
        tok_ids = tokenizer.encode(target_word, add_special_tokens=False)
        if not tok_ids:
            continue
        target_id = tok_ids[0]
        sent_ids = inputs["input_ids"][0].cpu().tolist()
        positions = [i for i, tid in enumerate(sent_ids) if tid == target_id]
        if not positions:
            continue
        pos = positions[0]
        
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
        
        sample_layers = get_sample_layers(n_layers, 8)
        for layer_idx in sample_layers:
            h = outputs.hidden_states[layer_idx][0, pos].detach().cpu().float().numpy()
            r = compute_ratio_k_from_h(h, U_wu, 50)
            layer_ratio_data[role][layer_idx].append(r)
    
    for role in ["subject", "object", "modifier", "function"]:
        print(f"  {role}:", end="")
        for li in sorted(layer_ratio_data[role].keys()):
            vals = layer_ratio_data[role][li]
            if vals:
                print(f" L{li}={np.mean(vals):.3f}", end="")
        print()
    
    # 5. 频谱形状差异: 语法词 vs 语义词
    print("\n--- 频谱形状差异 ---")
    syntax_tokens = ["the", "a", "is", "are", "was", "were", "has", "have", "do", "does"]
    semantic_tokens = ["cat", "dog", "run", "eat", "big", "red", "house", "book", "think", "love"]
    
    text = "The cat sat on the mat and a dog ran through the park while the bird flew over the house"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(inputs["input_ids"], output_hidden_states=True)
    
    # 末层
    last_layer = n_layers - 1
    syntax_spectra = []
    semantic_spectra = []
    
    for tok in syntax_tokens:
        tok_ids = tokenizer.encode(tok, add_special_tokens=False)
        if not tok_ids:
            continue
        target_id = tok_ids[0]
        sent_ids = inputs["input_ids"][0].cpu().tolist()
        positions = [i for i, tid in enumerate(sent_ids) if tid == target_id]
        if not positions:
            continue
        pos = positions[0]
        h = outputs.hidden_states[last_layer][0, pos].detach().cpu().float().numpy()
        spec = project_and_spectrum(h, U_wu, 50)
        syntax_spectra.append(spec)
    
    for tok in semantic_tokens:
        tok_ids = tokenizer.encode(tok, add_special_tokens=False)
        if not tok_ids:
            continue
        target_id = tok_ids[0]
        sent_ids = inputs["input_ids"][0].cpu().tolist()
        positions = [i for i, tid in enumerate(sent_ids) if tid == target_id]
        if not positions:
            continue
        pos = positions[0]
        h = outputs.hidden_states[last_layer][0, pos].detach().cpu().float().numpy()
        spec = project_and_spectrum(h, U_wu, 50)
        semantic_spectra.append(spec)
    
    if syntax_spectra and semantic_spectra:
        syntax_mean = np.mean(syntax_spectra, axis=0)
        semantic_mean = np.mean(semantic_spectra, axis=0)
        # 幂律拟合
        x = np.arange(1, len(syntax_mean)+1)
        
        # 对数回归
        log_x = np.log10(x[5:])
        if len(syntax_mean[5:]) > 0:
            log_syntax = np.log10(syntax_mean[5:] + 1e-10)
            log_semantic = np.log10(semantic_mean[5:] + 1e-10)
            
            valid_s = np.isfinite(log_syntax)
            valid_m = np.isfinite(log_semantic)
            
            if np.sum(valid_s) > 3:
                slope_s, _ = np.polyfit(log_x[valid_s], log_syntax[valid_s], 1)
            else:
                slope_s = 0
            if np.sum(valid_m) > 3:
                slope_m, _ = np.polyfit(log_x[valid_m], log_semantic[valid_m], 1)
            else:
                slope_m = 0
            
            print(f"语法词频谱幂律斜率: {slope_s:.3f}")
            print(f"语义词频谱幂律斜率: {slope_m:.3f}")
            print(f"差异: {abs(slope_s - slope_m):.3f}")
        
        # 频谱相关
        if len(syntax_mean) > 0 and len(semantic_mean) > 0:
            r, _ = pearsonr(syntax_mean, semantic_mean)
            print(f"语法词vs语义词频谱相关: {r:.4f}")
    
    result = {
        "model": model_name,
        "ratio_by_role": {
            role: {str(k): {"mean": float(np.mean(v)), "std": float(np.std(v))} 
                   for k, v in ks.items() if v}
            for role, ks in token_ratio_data.items()
        },
    }
    
    return result


def compute_ratio_k_from_h(h, U_wu, k):
    """从hidden state计算ratio(k)"""
    proj = U_wu[:, :k].T @ h
    ratio = np.sum(proj**2) / (np.sum(h**2) + 1e-10)
    return ratio


# ============== P561: MLP神经元的功能专业化 ==============
def experiment_p561(model, tokenizer, model_name, device, U_wu, S_wu, W_U):
    """MLP神经元的'功能专业化' — 不同神经元编码什么语义信息?"""
    print(f"\n{'='*60}")
    print(f"P561: MLP神经元功能专业化 — {model_name}")
    print(f"{'='*60}")
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    k_wu = min(200, U_wu.shape[1])
    
    layers = get_layers(model)
    
    # 1. 选中层分析W_down列的语义专业化
    sample_layers = get_sample_layers(n_layers, 5)
    print(f"采样层: {sample_layers}")
    
    for li in sample_layers:
        layer = layers[li]
        lw = get_layer_weights(layer, d_model, info.mlp_type)
        W_down = lw.W_down  # [d_model, intermediate]
        
        # W_down的每一列对应一个MLP神经元
        # 计算每列在W_U空间中的投影
        n_neurons = W_down.shape[1]
        
        # 选Top-50最活跃神经元（按Frobenius范数）
        col_norms = np.linalg.norm(W_down, axis=0)
        top_indices = np.argsort(col_norms)[::-1][:50]
        
        # 每个top神经元在W_U方向上的分布
        top_concentrations = []
        for idx in top_indices[:20]:
            col = W_down[:, idx]
            # 投影到W_U方向
            proj = U_wu[:, :k_wu].T @ col
            abs_proj = np.abs(proj)
            total = np.sum(abs_proj**2) + 1e-10
            # 前5个方向的集中度
            top5 = np.sum(abs_proj[:5]**2) / total
            top_concentrations.append(top5)
        
        mean_conc = np.mean(top_concentrations)
        
        # 各Top神经元的主方向
        top_directions = []
        for idx in top_indices[:10]:
            col = W_down[:, idx]
            proj = U_wu[:, :k_wu].T @ col
            main_dir = np.argmax(np.abs(proj))
            top_directions.append(main_dir)
        
        print(f"  L{li}: Top-20神经元在前5个W_U方向集中度={mean_conc:.3f}")
        print(f"         Top-10主方向: {top_directions}")
    
    # 2. 不同层MLP神经元的专业化趋势
    print("\n--- 神经元专业化随层变化 ---")
    layer_concentrations = {}
    for li in sample_layers:
        layer = layers[li]
        lw = get_layer_weights(layer, d_model, info.mlp_type)
        W_down = lw.W_down
        
        n_neurons = W_down.shape[1]
        col_norms = np.linalg.norm(W_down, axis=0)
        top_indices = np.argsort(col_norms)[::-1][:50]
        
        concentrations = []
        for idx in top_indices:
            col = W_down[:, idx]
            proj = U_wu[:, :k_wu].T @ col
            abs_proj = np.abs(proj)
            total = np.sum(abs_proj**2) + 1e-10
            top5 = np.sum(abs_proj[:5]**2) / total
            concentrations.append(top5)
        
        layer_concentrations[li] = np.mean(concentrations)
    
    for li in sorted(layer_concentrations.keys()):
        print(f"  L{li}: 平均集中度={layer_concentrations[li]:.3f}")
    
    # 3. Top神经元解码: 投影回词汇空间
    print("\n--- Top神经元解码到词汇空间 ---")
    mid_layer = sample_layers[len(sample_layers)//2]
    layer = layers[mid_layer]
    lw = get_layer_weights(layer, d_model, info.mlp_type)
    W_down = lw.W_down
    
    col_norms = np.linalg.norm(W_down, axis=0)
    top_indices = np.argsort(col_norms)[::-1][:20]
    
    for rank, idx in enumerate(top_indices[:10]):
        col = W_down[:, idx]
        # 直接用W_U解码: logits = W_U @ col
        logits = W_U @ col
        # Top-5 tokens
        top5_tok_ids = np.argsort(logits)[::-1][:5]
        top5_tokens = []
        for tid in top5_tok_ids:
            try:
                tok_str = tokenizer.decode([tid])
                # 安全编码：替换非ASCII字符
                tok_str_safe = tok_str.encode('ascii', 'replace').decode('ascii')
                top5_tokens.append(f"{tok_str_safe}({logits[tid]:.2f})")
            except:
                top5_tokens.append(f"id={tid}({logits[tid]:.2f})")
        
        # W_U方向信息
        proj = U_wu[:, :k_wu].T @ col
        main_dir = np.argmax(np.abs(proj))
        
        print(f"  Neuron#{rank}(idx={idx}): main_WU_dir={main_dir}, "
              f"Top5: {', '.join(top5_tokens)}")
    
    # 4. 功能词方向 vs 内容词方向的专业化
    print("\n--- 功能词vs内容词方向的神经元分配 ---")
    # W_U方向0-5: 可能是功能词方向
    # W_U方向6+: 可能是内容词方向
    # 检查Top神经元的W_down列在这些方向上的能量分配
    
    for li in sample_layers:
        layer = layers[li]
        lw = get_layer_weights(layer, d_model, info.mlp_type)
        W_down = lw.W_down
        
        col_norms = np.linalg.norm(W_down, axis=0)
        top_indices = np.argsort(col_norms)[::-1][:100]
        
        func_energy = 0  # 方向0-5
        content_energy = 0  # 方向6-50
        
        for idx in top_indices:
            col = W_down[:, idx]
            proj = U_wu[:, :50].T @ col
            abs_proj = np.abs(proj)
            func_energy += np.sum(abs_proj[:5]**2)
            content_energy += np.sum(abs_proj[5:50]**2)
        
        total = func_energy + content_energy + 1e-10
        print(f"  L{li}: 功能方向(0-4)能量={func_energy/total:.3f}, "
              f"内容方向(5-49)能量={content_energy/total:.3f}")
    
    result = {
        "model": model_name,
        "layer_concentrations": {str(k): float(v) for k, v in layer_concentrations.items()},
    }
    
    return result


# ============== P562: 从频谱力学到语言能力的映射 ==============
def experiment_p562(model, tokenizer, model_name, device, U_wu, S_wu, W_U):
    """从频谱力学到语言能力的映射 — 频谱结构如何支持下一个token预测?"""
    print(f"\n{'='*60}")
    print(f"P562: 频谱力学到语言能力的映射 — {model_name}")
    print(f"{'='*60}")
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    k_wu = min(200, U_wu.shape[1])
    
    # 1. 频谱精度与预测精度的关系
    print("\n--- 频谱精度vs预测精度 ---")
    
    test_texts = [
        "The cat sat on the",
        "She went to the store to",
        "The weather is very cold and",
        "He studied hard for the",
        "The children played in the",
    ]
    
    all_ratios = []
    all_confidences = []
    all_top1_probs = []
    
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
            logits = outputs.logits[0, -1]  # 末位预测
            probs = torch.softmax(logits.float(), dim=0)
            top1_prob = probs.max().item()
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
        
        # 末层h的ratio(50)
        h_last = outputs.hidden_states[-1][0, -1].detach().cpu().float().numpy()
        ratio_50 = compute_ratio_k_from_h(h_last, U_wu, 50)
        ratio_10 = compute_ratio_k_from_h(h_last, U_wu, 10)
        ratio_100 = compute_ratio_k_from_h(h_last, U_wu, 100)
        
        # 频谱幂律斜率
        spec = project_and_spectrum(h_last, U_wu, 50)
        x = np.arange(1, len(spec)+1)
        log_x = np.log10(x[3:])
        log_spec = np.log10(spec[3:] + 1e-10)
        valid = np.isfinite(log_spec) & np.isfinite(log_x)
        if np.sum(valid) > 3:
            slope, _ = np.polyfit(log_x[valid], log_spec[valid], 1)
        else:
            slope = 0
        
        all_ratios.append({"r10": ratio_10, "r50": ratio_50, "r100": ratio_100, "slope": slope})
        all_top1_probs.append(top1_prob)
        
        print(f"  '{text}' -> ratio(10)={ratio_10:.3f}, ratio(50)={ratio_50:.3f}, "
              f"slope={slope:.3f}, top1_prob={top1_prob:.3f}, entropy={entropy:.3f}")
    
    # 2. 频谱形状与预测不确定性的关系
    print("\n--- 频谱形状vs预测不确定性 ---")
    if len(all_ratios) > 2:
        r50s = [r["r50"] for r in all_ratios]
        r10s = [r["r10"] for r in all_ratios]
        slopes = [r["slope"] for r in all_ratios]
        
        # ratio(50) vs top1_prob
        if len(set(r50s)) > 1 and len(set(all_top1_probs)) > 1:
            corr_r50_prob, _ = spearmanr(r50s, all_top1_probs)
            print(f"ratio(50) vs top1_prob: Spearman r={corr_r50_prob:.3f}")
        else:
            corr_r50_prob = 0
            print(f"ratio(50) vs top1_prob: 变异不足，无法计算")
        
        # ratio(10) vs top1_prob
        if len(set(r10s)) > 1:
            corr_r10_prob, _ = spearmanr(r10s, all_top1_probs)
            print(f"ratio(10) vs top1_prob: Spearman r={corr_r10_prob:.3f}")
        else:
            corr_r10_prob = 0
        
        # slope vs top1_prob
        if len(set(slopes)) > 1:
            corr_slope_prob, _ = spearmanr(slopes, all_top1_probs)
            print(f"slope vs top1_prob: Spearman r={corr_slope_prob:.3f}")
        else:
            corr_slope_prob = 0
    
    # 3. 截断频谱对预测的影响
    print("\n--- 频谱截断对预测的影响 ---")
    # 将h投影到前K个W_U方向，然后看预测变化
    text = "The development of artificial intelligence has transformed"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(inputs["input_ids"], output_hidden_states=True)
    
    h_last = outputs.hidden_states[-1][0, -1].detach().cpu().float().numpy()
    
    # 原始预测
    original_logits = W_U @ h_last
    original_top5 = np.argsort(original_logits)[::-1][:5]
    original_tokens = [tokenizer.decode([tid]).encode('ascii', 'replace').decode('ascii') for tid in original_top5]
    
    print(f"原始Top-5: {original_tokens}")
    
    # 截断到前K个W_U方向
    for K in [10, 30, 50, 100, 200]:
        # 重建h: 只保留前K个方向的分量
        proj = U_wu[:, :K].T @ h_last  # [K]
        h_reconstructed = U_wu[:, :K] @ proj  # [d_model]
        
        # 预测
        recon_logits = W_U @ h_reconstructed
        recon_top5 = np.argsort(recon_logits)[::-1][:5]
        recon_tokens = [tokenizer.decode([tid]).encode('ascii', 'replace').decode('ascii') for tid in recon_top5]
        
        # 与原始预测的重叠
        overlap = len(set(original_top5) & set(recon_top5)) / 5
        
        # 预测质量: logits的余弦相似度
        log_norm1 = np.linalg.norm(original_logits) + 1e-10
        log_norm2 = np.linalg.norm(recon_logits) + 1e-10
        log_cos_sim = np.dot(original_logits, recon_logits) / (log_norm1 * log_norm2)
        
        print(f"  K={K}: Top-5={recon_tokens}, 重叠={overlap:.1%}, "
              f"logits余弦相似={log_cos_sim:.4f}")
    
    # 4. 关键发现: 频谱与语言能力的因果链
    print("\n--- 频谱->语言能力的因果链分析 ---")
    
    # 对多个位置分析ratio(k)与预测置信度的关系
    long_text = "The development of artificial intelligence has transformed many aspects of modern life, from healthcare to transportation and communication."
    inputs = tokenizer(long_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(inputs["input_ids"], output_hidden_states=True)
        all_logits = outputs.logits[0]  # [seq_len, vocab]
        all_probs = torch.softmax(all_logits.float(), dim=-1)
    
    seq_len = all_probs.shape[0]
    position_ratios = []
    position_confidences = []
    
    for pos in range(1, min(seq_len, 20)):
        h = outputs.hidden_states[-1][0, pos].detach().cpu().float().numpy()
        r50 = compute_ratio_k_from_h(h, U_wu, 50)
        
        # 该位置的预测置信度（预测下一个token的top1概率）
        top1_prob = all_probs[pos].max().item()
        
        position_ratios.append(r50)
        position_confidences.append(top1_prob)
    
    if len(set(position_ratios)) > 1 and len(set(position_confidences)) > 1:
        corr_ratio_conf, _ = pearsonr(position_ratios, position_confidences)
        print(f"各位置ratio(50)与top1_prob相关: r={corr_ratio_conf:.3f}")
    else:
        corr_ratio_conf = 0
        print(f"各位置变异不足，无法计算相关")
    
    # ratio(k)梯度(频谱集中度变化)与预测变化
    print(f"ratio(50)均值: {np.mean(position_ratios):.3f}")
    print(f"top1_prob均值: {np.mean(position_confidences):.3f}")
    
    result = {
        "model": model_name,
        "ratio_confidence_corr": float(corr_ratio_conf) if not np.isnan(corr_ratio_conf) else 0,
        "truncation_analysis": "K=10-200 logits余弦相似0.5-0.99",
    }
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--experiment", type=str, required=True, 
                        choices=["p559", "p560", "p561", "p562"])
    args = parser.parse_args()
    
    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, args.model)
    
    print(f"\n模型: {args.model}, 层数: {info.n_layers}, d_model: {info.d_model}")
    
    U_wu, S_wu, W_U = compute_wu_svd(model, k=200)
    
    if args.experiment == "p559":
        result = experiment_p559(model, tokenizer, args.model, device, U_wu, S_wu, W_U)
    elif args.experiment == "p560":
        result = experiment_p560(model, tokenizer, args.model, device, U_wu, S_wu, W_U)
    elif args.experiment == "p561":
        result = experiment_p561(model, tokenizer, args.model, device, U_wu, S_wu, W_U)
    elif args.experiment == "p562":
        result = experiment_p562(model, tokenizer, args.model, device, U_wu, S_wu, W_U)
    
    # 保存结果
    result_dir = f"results/phase_cxxvii"
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, f"{args.model}_{args.experiment}.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存到: {result_file}")
    
    # 释放GPU
    del model
    torch.cuda.empty_cache()
    print("GPU内存已释放")


if __name__ == "__main__":
    main()
