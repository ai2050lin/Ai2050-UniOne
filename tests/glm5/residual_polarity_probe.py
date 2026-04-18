"""
Phase CLXXXVII: 残差流逐维度极性解码 — 找到极性信息编码的维度
============================================================

核心问题: Phase CLXXXV-VI发现:
  1. 1维极性方向注入无因果效应(斜率<0.05)
  2. 完整残差修补有效(100-300x)
  3. 注意力头分布式编码(无关键头)
  → 极性信息编码在高维表示中, 但具体在哪些维度?

测试设计:
D1: Linear Probe — 从残差流解码极性(neg vs aff), 逐层评估
    → 找到极性信息最强的层
D2: 维度重要性分析 — 对每一层, 用probe权重找最重要的维度
    → 极性信息集中在多少个维度上?
D3: 逐维度Ablation — 关闭单个维度, 看极性解码性能下降
    → 因果验证: 哪些维度是极性必需的?
D4: 跨层维度追踪 — 极性维度在不同层之间是否一致?
    → 极性信息如何在层间传播?

运行:
  python tests/glm5/residual_polarity_probe.py --model qwen3 --test d1
  python tests/glm5/residual_polarity_probe.py --model qwen3 --test all
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import gc
import time
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


# ============================================================
# 测试集 — 更多样本以提高probe可靠性
# ============================================================

POLARITY_PAIRS = [
    # is not
    ("The cat is here", "The cat is not here"),
    ("The dog is happy", "The dog is not happy"),
    ("The house is big", "The house is not big"),
    ("The phone is working", "The phone is not working"),
    ("The bridge is safe", "The bridge is not safe"),
    ("The star is visible", "The star is not visible"),
    ("The door was closed", "The door was not closed"),
    ("The child was playing", "The child was not playing"),
    # does not
    ("I like the car", "I do not like the car"),
    ("She knows the answer", "She does not know the answer"),
    ("The river flows north", "The river does not flow north"),
    ("I understand the plan", "I do not understand the plan"),
    ("She likes the movie", "She does not like the movie"),
    ("The key works well", "The key does not work well"),
    ("The food tastes good", "The food does not taste good"),
    # can/will not
    ("He can swim", "He cannot swim"),
    ("The bird will come", "The bird will not come"),
    # did not
    ("The cloud disappeared", "The cloud did not disappear"),
    # has not
    ("The flower has bloomed", "The flower has not bloomed"),
    # extra
    ("The light shines bright", "The light does not shine bright"),
    ("The wind blows cold", "The wind does not blow cold"),
    ("The rain falls down", "The rain does not fall down"),
    ("The sun rises early", "The sun does not rise early"),
    ("The table holds weight", "The table does not hold weight"),
]

# 额外测试对 — 用于验证probe泛化性
EXTRA_PAIRS = [
    ("The water is clean", "The water is not clean"),
    ("The machine works", "The machine does not work"),
    ("He can see", "He cannot see"),
    ("The plan was approved", "The plan was not approved"),
    ("She will come", "She will not come"),
    ("The road is open", "The road is not open"),
    ("The system is stable", "The system is not stable"),
    ("I trust the result", "I do not trust the result"),
]


# ============================================================
# 模型加载
# ============================================================

def load_model_fast(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    PATHS = {
        "qwen3": "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
        "glm4": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
        "deepseek7b": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    }
    
    path = PATHS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if model_name in ["glm4", "deepseek7b"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            path, quantization_config=bnb_config, device_map="auto",
            trust_remote_code=True, local_files_only=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map='cpu',
            trust_remote_code=True, local_files_only=True,
        )
        model = model.to('cuda')
    
    model.eval()
    device = next(model.parameters()).device
    return model, tokenizer, device


# ============================================================
# 核心工具: 提取残差流
# ============================================================

def extract_residual_stream(model, tokenizer, device, texts, layers_to_probe, pool_method='last'):
    """
    提取残差流表示
    
    Args:
        model: 模型
        tokenizer: 分词器
        device: 设备
        texts: 文本列表
        layers_to_probe: 要提取的层列表
        pool_method: 'last'=最后位置, 'mean'=均值, 'not_pos'='not'位置
    
    Returns:
        dict: {layer_idx: np.array [n_texts, d_model]}
    """
    captured = {}
    hooks = []
    
    layers = model.model.layers
    
    for li in layers_to_probe:
        captured[li] = []
    
    def make_hook(li):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0].detach().cpu().float()
            else:
                h = output.detach().cpu().float()
            captured[li].append(h[0])  # [seq_len, d_model]
        return hook_fn
    
    # 注册hooks
    for li in layers_to_probe:
        hooks.append(layers[li].register_forward_hook(make_hook(li)))
    
    results = {li: [] for li in layers_to_probe}
    
    for text in texts:
        # 清空captured
        for li in layers_to_probe:
            captured[li] = []
        
        toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
        
        with torch.no_grad():
            try:
                _ = model(**toks)
            except Exception as e:
                print(f"  Forward failed for '{text}': {e}")
                for li in layers_to_probe:
                    results[li].append(np.zeros(model.config.hidden_size))
                continue
        
        for li in layers_to_probe:
            if captured[li]:
                h = captured[li][0].numpy()  # [seq_len, d_model]
                
                if pool_method == 'last':
                    results[li].append(h[-1])
                elif pool_method == 'mean':
                    results[li].append(h.mean(axis=0))
                elif pool_method == 'not_pos':
                    # Find 'not' position
                    input_ids = toks.input_ids[0].tolist()
                    not_pos = None
                    for i, tid in enumerate(input_ids):
                        if tokenizer.decode([tid]).strip().lower() == 'not':
                            not_pos = i
                            break
                    if not_pos is not None and not_pos < len(h):
                        results[li].append(h[not_pos])
                    else:
                        results[li].append(h[-1])  # fallback
                else:
                    results[li].append(h[-1])
            else:
                results[li].append(np.zeros(model.config.hidden_size))
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    # Convert to numpy arrays
    for li in layers_to_probe:
        results[li] = np.array(results[li])  # [n_texts, d_model]
    
    return results


# ============================================================
# D1: Linear Probe — 逐层解码极性
# ============================================================

def test_d1(model_name, model, tokenizer, device, d_model, n_layers):
    """D1: Linear Probe — 从残差流解码极性(neg vs aff)"""
    
    print("\n" + "=" * 70)
    print("D1: Linear Probe — 逐层残差流极性解码")
    print("=" * 70)
    
    # 准备数据
    aff_texts = [aff for aff, neg in POLARITY_PAIRS]
    neg_texts = [neg for aff, neg in POLARITY_PAIRS]
    all_texts = aff_texts + neg_texts
    labels = [0] * len(aff_texts) + [1] * len(neg_texts)
    
    print(f"  训练集: {len(aff_texts)}肯定 + {len(neg_texts)}否定 = {len(all_texts)}")
    
    # 额外测试集
    extra_aff = [aff for aff, neg in EXTRA_PAIRS]
    extra_neg = [neg for aff, neg in EXTRA_PAIRS]
    extra_texts = extra_aff + extra_neg
    extra_labels = [0] * len(extra_aff) + [1] * len(extra_neg)
    
    # 采样层
    sample_layers = list(range(0, n_layers, max(1, n_layers // 8))) + [n_layers - 1]
    sample_layers = sorted(set(sample_layers))
    
    print(f"  采样层: {sample_layers}")
    
    # 对不同pooling方法做测试
    pool_methods = ['last', 'mean', 'not_pos']
    
    results = {
        'sample_layers': sample_layers,
        'n_train': len(all_texts),
        'n_test': len(extra_texts),
        'probe_accuracy': {},  # {(pool, layer): accuracy}
        'probe_cross_val': {},  # {(pool, layer): cv_score}
    }
    
    for pool in pool_methods:
        print(f"\n  [D1] Pool method: {pool}")
        
        # 提取训练集表示
        print(f"  提取训练集残差流...")
        train_repr = extract_residual_stream(model, tokenizer, device, all_texts, sample_layers, pool)
        
        # 提取测试集表示
        print(f"  提取测试集残差流...")
        test_repr = extract_residual_stream(model, tokenizer, device, extra_texts, sample_layers, pool)
        
        for li in sample_layers:
            X_train = train_repr[li]
            X_test = test_repr[li]
            y_train = np.array(labels)
            y_test = np.array(extra_labels)
            
            # 标准化
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Linear Probe (Logistic Regression)
            clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
            
            # Cross-validation on training set
            try:
                cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=min(5, len(all_texts)//4), scoring='accuracy')
                cv_mean = np.mean(cv_scores)
            except:
                cv_mean = 0.0
            
            # Train on full training set, test on extra set
            try:
                clf.fit(X_train_scaled, y_train)
                test_acc = clf.score(X_test_scaled, y_test)
            except:
                test_acc = 0.0
            
            results['probe_accuracy'][(pool, li)] = float(test_acc)
            results['probe_cross_val'][(pool, li)] = float(cv_mean)
            
            marker = " *" if test_acc > 0.8 or cv_mean > 0.9 else ""
            print(f"    L{li:2d}: CV={cv_mean:.3f}, Test={test_acc:.3f}{marker}")
    
    # 找到最佳层和pooling
    print("\n  [D1] 最佳层/pooling:")
    for pool in pool_methods:
        best_li = max(sample_layers, key=lambda li: results['probe_accuracy'].get((pool, li), 0))
        best_acc = results['probe_accuracy'].get((pool, best_li), 0)
        best_cv = results['probe_cross_val'].get((pool, best_li), 0)
        print(f"    {pool}: Best=L{best_li}, CV={best_cv:.3f}, Test={best_acc:.3f}")
    
    return results


# ============================================================
# D2: 维度重要性分析
# ============================================================

def test_d2(model_name, model, tokenizer, device, d_model, n_layers):
    """D2: 维度重要性分析 — 极性信息集中在多少个维度上?"""
    
    print("\n" + "=" * 70)
    print("D2: 维度重要性分析 — 极性维度集中度")
    print("=" * 70)
    
    # 准备数据
    aff_texts = [aff for aff, neg in POLARITY_PAIRS]
    neg_texts = [neg for aff, neg in POLARITY_PAIRS]
    all_texts = aff_texts + neg_texts
    labels = np.array([0] * len(aff_texts) + [1] * len(neg_texts))
    
    # 关键层: 基于D1的结果
    key_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    key_layers = sorted(set(key_layers))
    
    print(f"  关键层: {key_layers}")
    print(f"  d_model: {d_model}")
    
    results = {
        'key_layers': key_layers,
        'dim_importance': {},  # {layer: {dim_idx: importance}}
        'top_dims': {},  # {layer: [top_k_dim_indices]}
        'cumulative_importance': {},  # {layer: {k: cumulative_importance}}
    }
    
    for pool in ['last', 'not_pos']:
        print(f"\n  [D2] Pool: {pool}")
        
        repr_dict = extract_residual_stream(model, tokenizer, device, all_texts, key_layers, pool)
        
        for li in key_layers:
            X = repr_dict[li]
            
            # 标准化
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Logistic Regression
            clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
            try:
                clf.fit(X_scaled, labels)
            except:
                print(f"    L{li}: Probe failed")
                continue
            
            # 权重重要性: |w_i| (对d_model个维度)
            w = clf.coef_[0]  # [d_model]
            importance = np.abs(w)
            
            # 排序
            sorted_dims = np.argsort(importance)[::-1]
            sorted_importance = importance[sorted_dims]
            
            # 累积重要性
            total_importance = np.sum(sorted_importance)
            cumulative = np.cumsum(sorted_importance) / total_importance
            
            # 找到90%和50%累积重要性需要的维度数
            dim_50 = int(np.searchsorted(cumulative, 0.5)) + 1
            dim_90 = int(np.searchsorted(cumulative, 0.9)) + 1
            dim_95 = int(np.searchsorted(cumulative, 0.95)) + 1
            
            # Top-10维度
            top10_dims = sorted_dims[:10].tolist()
            top10_importance = sorted_importance[:10].tolist()
            
            # 极性方向 vs PCA方向
            # 极性差分向量
            aff_mean = X[:len(aff_texts)].mean(axis=0)
            neg_mean = X[len(aff_texts):].mean(axis=0)
            polarity_direction = neg_mean - aff_mean
            polarity_direction = polarity_direction / (np.linalg.norm(polarity_direction) + 1e-10)
            
            # Probe权重方向
            w_direction = w / (np.linalg.norm(w) + 1e-10)
            
            # 两者的一致性
            cos_probe_polarity = float(np.dot(w_direction, polarity_direction))
            
            key = f"{pool}_L{li}"
            results['dim_importance'][key] = {
                'dim_50': int(dim_50),
                'dim_90': int(dim_90),
                'dim_95': int(dim_95),
                'cos_probe_polarity': float(cos_probe_polarity),
                'top10_dims': top10_dims,
                'top10_importance': [round(float(x), 6) for x in top10_importance],
                'total_importance': float(total_importance),
            }
            results['cumulative_importance'][key] = {
                'k10': float(cumulative[9]) if len(cumulative) >= 10 else 1.0,
                'k50': float(cumulative[49]) if len(cumulative) >= 50 else 1.0,
                'k100': float(cumulative[99]) if len(cumulative) >= 100 else 1.0,
                'k500': float(cumulative[499]) if len(cumulative) >= 500 else 1.0,
            }
            
            print(f"    L{li} ({pool}): dim_50={dim_50}, dim_90={dim_90}, dim_95={dim_95}, "
                  f"cos(probe,polarity)={cos_probe_polarity:+.3f}, "
                  f"top10_cum={cumulative[9]:.3f}")
    
    return results


# ============================================================
# D3: 逐维度Ablation — 因果验证
# ============================================================

def test_d3(model_name, model, tokenizer, device, d_model, n_layers):
    """D3: 逐维度Ablation — 关闭单个维度, 看极性解码性能下降"""
    
    print("\n" + "=" * 70)
    print("D3: 逐维度Ablation — 因果验证极性维度")
    print("=" * 70)
    
    # 准备数据
    aff_texts = [aff for aff, neg in POLARITY_PAIRS]
    neg_texts = [neg for aff, neg in POLARITY_PAIRS]
    all_texts = aff_texts + neg_texts
    labels = np.array([0] * len(aff_texts) + [1] * len(neg_texts))
    
    not_tok_id = None
    test_ids = tokenizer.encode("The cat is not here", add_special_tokens=False)
    for tid in test_ids:
        if tokenizer.decode([tid]).strip().lower() == "not":
            not_tok_id = tid
            break
    
    # 只在最关键的一层做逐维度ablation
    target_layer = n_layers // 2  # 中间层
    print(f"  目标层: L{target_layer}")
    
    # 先训练baseline probe
    repr_all = extract_residual_stream(model, tokenizer, device, all_texts, [target_layer], 'last')
    X_base = repr_all[target_layer]
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_base)
    
    clf_base = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    try:
        clf_base.fit(X_scaled, labels)
    except:
        print("  Baseline probe failed")
        return {}
    
    # Baseline: cross-validated accuracy
    base_cv = np.mean(cross_val_score(clf_base, X_scaled, labels, cv=5, scoring='accuracy'))
    base_w = clf_base.coef_[0].copy()  # [d_model]
    
    print(f"  Baseline CV accuracy: {base_cv:.3f}")
    
    # 按probe权重重要性排序
    importance = np.abs(base_w)
    sorted_dims = np.argsort(importance)[::-1]
    
    # 逐维度ablation: 一次关闭多个维度(前1, 前5, 前10, 前50, 前100)
    # 因为一次关闭1个维度效果太弱
    ablation_counts = [1, 5, 10, 20, 50, 100, 200, 500, d_model]
    
    results = {
        'target_layer': target_layer,
        'base_cv': float(base_cv),
        'ablation_results': [],
    }
    
    # 也测试: 关闭随机维度作为对照
    np.random.seed(42)
    random_dims = np.random.choice(d_model, size=d_model, replace=False)
    
    print(f"\n  逐维度Ablation (按重要性排序):")
    
    for n_ablate in ablation_counts:
        if n_ablate > d_model:
            n_ablate = d_model
        
        # 关闭前n_ablate个重要维度
        dims_to_ablate = sorted_dims[:n_ablate]
        
        X_ablated = X_base.copy()
        X_ablated[:, dims_to_ablate] = 0
        
        X_ablated_scaled = scaler.transform(X_ablated)
        
        try:
            cv_ablated = np.mean(cross_val_score(
                LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs'),
                X_ablated_scaled, labels, cv=5, scoring='accuracy'))
        except:
            cv_ablated = 0.0
        
        # 关闭随机n_ablate个维度
        random_dims_to_ablate = random_dims[:n_ablate]
        X_random_ablated = X_base.copy()
        X_random_ablated[:, random_dims_to_ablate] = 0
        X_random_scaled = scaler.transform(X_random_ablated)
        
        try:
            cv_random = np.mean(cross_val_score(
                LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs'),
                X_random_scaled, labels, cv=5, scoring='accuracy'))
        except:
            cv_random = 0.0
        
        results['ablation_results'].append({
            'n_ablate': int(n_ablate),
            'important_cv': float(cv_ablated),
            'random_cv': float(cv_random),
            'important_drop': float(base_cv - cv_ablated),
            'random_drop': float(base_cv - cv_random),
            'selectivity': float((base_cv - cv_ablated) / max(base_cv - cv_random, 1e-6)),
        })
        
        sel = (base_cv - cv_ablated) / max(base_cv - cv_random, 1e-6)
        print(f"    n={n_ablate:4d}: important_cv={cv_ablated:.3f}(drop={base_cv-cv_ablated:.3f}), "
              f"random_cv={cv_random:.3f}(drop={base_cv-cv_random:.3f}), sel={sel:.1f}x")
    
    # 关键测试: 关闭前10个重要维度 vs 前10个随机维度
    print(f"\n  关键对比 (Top-10重要 vs 随机10):")
    top10_imp = results['ablation_results'][2]  # n=10
    print(f"    重要10维: drop={top10_imp['important_drop']:.3f}")
    print(f"    随机10维: drop={top10_imp['random_drop']:.3f}")
    print(f"    选择性: {top10_imp['selectivity']:.1f}x")
    
    return results


# ============================================================
# D4: 跨层维度追踪
# ============================================================

def test_d4(model_name, model, tokenizer, device, d_model, n_layers):
    """D4: 跨层维度追踪 — 极性维度在不同层之间是否一致?"""
    
    print("\n" + "=" * 70)
    print("D4: 跨层维度追踪 — 极性维度的层间一致性")
    print("=" * 70)
    
    # 准备数据
    aff_texts = [aff for aff, neg in POLARITY_PAIRS]
    neg_texts = [neg for aff, neg in POLARITY_PAIRS]
    all_texts = aff_texts + neg_texts
    labels = np.array([0] * len(aff_texts) + [1] * len(neg_texts))
    
    sample_layers = list(range(0, n_layers, max(1, n_layers // 6))) + [n_layers - 1]
    sample_layers = sorted(set(sample_layers))
    
    print(f"  采样层: {sample_layers}")
    
    # 提取各层表示
    repr_dict = extract_residual_stream(model, tokenizer, device, all_texts, sample_layers, 'last')
    
    # 训练每层的probe
    from sklearn.preprocessing import StandardScaler
    
    layer_probes = {}
    layer_top_dims = {}
    
    for li in sample_layers:
        X = repr_dict[li]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
        try:
            clf.fit(X_scaled, labels)
        except:
            continue
        
        w = clf.coef_[0]
        importance = np.abs(w)
        top_dims = set(np.argsort(importance)[-50:].tolist())  # Top-50 important dims
        
        layer_probes[li] = w
        layer_top_dims[li] = top_dims
    
    # 计算层间一致性
    print(f"\n  层间维度重叠 (Top-50 dims):")
    
    results = {
        'sample_layers': sample_layers,
        'dim_overlap': {},
        'probe_correlation': {},
    }
    
    for i, li1 in enumerate(sample_layers):
        for li2 in sample_layers[i+1:]:
            if li1 not in layer_top_dims or li2 not in layer_top_dims:
                continue
            
            # Jaccard overlap
            overlap = len(layer_top_dims[li1] & layer_top_dims[li2])
            union = len(layer_top_dims[li1] | layer_top_dims[li2])
            jaccard = overlap / max(union, 1)
            
            # Cosine similarity of probe weights
            w1 = layer_probes[li1]
            w2 = layer_probes[li2]
            cos = float(np.dot(w1, w2) / (np.linalg.norm(w1) * np.linalg.norm(w2) + 1e-10))
            
            results['dim_overlap'][(li1, li2)] = float(jaccard)
            results['probe_correlation'][(li1, li2)] = float(cos)
            
            if abs(cos) > 0.3 or jaccard > 0.1:
                print(f"    L{li1}-L{li2}: Jaccard={jaccard:.3f}, cos_probe={cos:+.3f}")
    
    # 特殊对比: L0 vs 最终层 vs 中间层
    l0 = 0
    l_mid = n_layers // 2
    l_last = n_layers - 1
    
    for li2 in [l_mid, l_last]:
        if l0 in layer_top_dims and li2 in layer_top_dims:
            overlap = len(layer_top_dims[l0] & layer_top_dims[li2])
            jaccard = overlap / len(layer_top_dims[l0] | layer_top_dims[li2])
            cos = float(np.dot(layer_probes[l0], layer_probes[li2]) / 
                        (np.linalg.norm(layer_probes[l0]) * np.linalg.norm(layer_probes[li2]) + 1e-10))
            print(f"\n  L0 vs L{li2}: Jaccard={jaccard:.3f}, cos_probe={cos:+.3f}")
    
    return results


# ============================================================
# 主入口
# ============================================================

def run_test(model_name, test_name):
    print(f"\n{'='*70}")
    print(f"Phase CLXXXVII: Residual Polarity Probe - {model_name}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model_fast(model_name)
    n_layers = len(model.model.layers)
    d_model = model.config.hidden_size
    
    print(f"  n_layers={n_layers}, d_model={d_model}")
    
    result_dir = Path(f"results/residual_probe/{model_name}")
    result_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    results = {}
    
    if test_name in ['d1', 'all']:
        r = test_d1(model_name, model, tokenizer, device, d_model, n_layers)
        results['d1'] = r
        # Deep convert tuple keys to strings for JSON
        def convert_keys(obj):
            if isinstance(obj, dict):
                new_dict = {}
                for k, v in obj.items():
                    new_key = str(k) if isinstance(k, tuple) else k
                    new_dict[new_key] = convert_keys(v)
                return new_dict
            elif isinstance(obj, list):
                return [convert_keys(item) for item in obj]
            else:
                return obj
        r_save = convert_keys(r)
        with open(result_dir / "d1_results.json", 'w') as f:
            json.dump(r_save, f, indent=2, default=str)
    
    if test_name in ['d2', 'all']:
        r = test_d2(model_name, model, tokenizer, device, d_model, n_layers)
        results['d2'] = r
        with open(result_dir / "d2_results.json", 'w') as f:
            json.dump(r, f, indent=2, default=str)
    
    if test_name in ['d3', 'all']:
        r = test_d3(model_name, model, tokenizer, device, d_model, n_layers)
        results['d3'] = r
        with open(result_dir / "d3_results.json", 'w') as f:
            json.dump(r, f, indent=2, default=str)
    
    if test_name in ['d4', 'all']:
        r = test_d4(model_name, model, tokenizer, device, d_model, n_layers)
        results['d4'] = r
        with open(result_dir / "d4_results.json", 'w') as f:
            json.dump(r, f, indent=2, default=str)
    
    elapsed = time.time() - start_time
    print(f"\n  Total time: {elapsed:.1f}s")
    
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--test", type=str, default="all", choices=["d1", "d2", "d3", "d4", "all"])
    args = parser.parse_args()
    
    run_test(args.model, args.test)
