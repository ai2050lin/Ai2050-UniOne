"""
Phase CLXXXV: 非线性功能探测 — 超越线性对齐度
核心问题: 线性对齐度和因果干预都证明功能方向≈随机，但功能方向确实可提取。
假设: 功能信号可能通过非线性通路编码，线性方法无法检测。

测试:
P851: MLP probe分类 — 用2层MLP从残差流分类功能类型(5类)
P852: MLP vs 线性probe对比 — 非线性增益有多大?
P853: 逐层功能可解码性 — 哪些层的功能信号最可解码?
P854: 功能信号的非线性结构 — 用kernel PCA分析功能差异的非线性结构

Usage:
    python phase_clxxxv_nonlinear_probe.py --model glm4
    python phase_clxxxv_nonlinear_probe.py --model qwen3
    python phase_clxxxv_nonlinear_probe.py --model deepseek7b
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'glm5'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_utils import load_model, release_model
import torch


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x)


def get_function_pairs_expanded():
    """扩展功能对: 7对/类 × 5类 = 35对，每对2个文本 = 70个样本"""
    return {
        'syntax': [
            ("The cat sits", "Cats sit"),
            ("She runs fast", "She run fast"),
            ("The big cat sleeps", "The cat big sleeps"),
            ("I have eaten lunch", "I eat have lunch"),
            ("He is running home", "Him running is home"),
            ("A cat that runs fast", "Cat a that runs fast"),
            ("She will go home", "Will she go home"),
        ],
        'semantic': [
            ("The cat sits quietly", "The dog sits quietly"),
            ("She is very happy now", "She is very sad now"),
            ("The house is quite big", "The house is quite small"),
            ("He really loves music", "He really hates music"),
            ("The king ruled wisely", "The queen ruled wisely"),
            ("I bought some food", "I stole some food"),
            ("The bright sun rises", "The bright sun sets"),
        ],
        'style': [
            ("I am happy today", "I'm happy today"),
            ("Do not go there", "Don't go there"),
            ("It is very fine", "It's very fine"),
            ("She will come soon", "She'll come soon"),
            ("I have done well", "I've done well"),
            ("Cannot do this now", "Can't do this now"),
            ("We are here now", "We're here now"),
        ],
        'tense': [
            ("She walks slowly", "She walked slowly"),
            ("He runs every day", "He ran every day"),
            ("I go to school", "I went to school"),
            ("They see the bird", "They saw the bird"),
            ("We eat at home", "We ate at home"),
            ("She writes a letter", "She wrote a letter"),
            ("He thinks about it", "He thought about it"),
        ],
        'polarity': [
            ("She is happy now", "She is not happy now"),
            ("I like this book", "I don't like this book"),
            ("He can swim well", "He cannot swim well"),
            ("This is good work", "This is not good work"),
            ("She will come back", "She will not come back"),
            ("They have enough", "They have no enough"),
            ("We know the truth", "We do not know truth"),
        ],
    }


def get_hidden_states_batch(model, tokenizer, device, texts, layer_idx=0):
    """批量获取残差流"""
    all_hs = []
    for text in texts:
        tokens = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
        hs = to_numpy(outputs.hidden_states[layer_idx][0, -1])
        all_hs.append(hs)
    return np.array(all_hs)


def run_experiment(model_name):
    """运行CLXXXV实验"""
    print(f"\n{'='*60}", flush=True)
    print(f"Phase CLXXXV: Nonlinear Functional Probe - {model_name}", flush=True)
    print(f"{'='*60}", flush=True)
    
    model, tokenizer, device = load_model(model_name)
    d_model = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    
    pairs_dict = get_function_pairs_expanded()
    func_type_names = list(pairs_dict.keys())
    
    # === 准备数据 ===
    print(f"\nPreparing data...", flush=True)
    
    # 收集所有文本和标签
    all_texts = []
    all_labels = []
    all_pair_ids = []
    
    pair_id = 0
    for ft_idx, (func_type, pairs) in enumerate(pairs_dict.items()):
        for text_a, text_b in pairs:
            all_texts.extend([text_a, text_b])
            all_labels.extend([ft_idx, ft_idx])
            all_pair_ids.extend([pair_id, pair_id])
            pair_id += 1
    
    n_samples = len(all_texts)
    print(f"Total samples: {n_samples}, n_classes: {len(func_type_names)}", flush=True)
    
    # === P851: MLP probe分类 ===
    print(f"\n--- P851: MLP Probe Classification ---", flush=True)
    
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    
    # 测试多个层
    test_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    
    p851_results = {}
    p852_results = {}
    p853_results = {}
    
    for layer_idx in test_layers:
        print(f"\n  Layer {layer_idx}:", flush=True)
        
        # 获取该层的残差流
        X = get_hidden_states_batch(model, tokenizer, device, all_texts, layer_idx)
        y = np.array(all_labels)
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 1) 线性probe
        clf_linear = LogisticRegression(max_iter=2000, C=1.0)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores_linear = cross_val_score(clf_linear, X_scaled, y, cv=cv, scoring='accuracy')
        
        # 2) MLP probe (2层, 隐藏64)
        clf_mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=42, early_stopping=True)
        scores_mlp = cross_val_score(clf_mlp, X_scaled, y, cv=cv, scoring='accuracy')
        
        # 3) 更大的MLP (2层, 隐藏128+64)
        clf_mlp2 = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42, early_stopping=True)
        scores_mlp2 = cross_val_score(clf_mlp2, X_scaled, y, cv=cv, scoring='accuracy')
        
        # 4) 随机基线
        rng = np.random.default_rng(42)
        y_shuffle = y.copy()
        rng.shuffle(y_shuffle)
        scores_random = cross_val_score(clf_mlp, X_scaled, y_shuffle, cv=cv, scoring='accuracy')
        
        print(f"    Linear: {scores_linear.mean():.3f}±{scores_linear.std():.3f}", flush=True)
        print(f"    MLP(64): {scores_mlp.mean():.3f}±{scores_mlp.std():.3f}", flush=True)
        print(f"    MLP(128,64): {scores_mlp2.mean():.3f}±{scores_mlp2.std():.3f}", flush=True)
        print(f"    Random: {scores_random.mean():.3f}±{scores_random.std():.3f}", flush=True)
        print(f"    Nonlinear gain: {scores_mlp.mean()/max(scores_linear.mean(),0.01):.2f}x", flush=True)
        
        p851_results[f'layer_{layer_idx}'] = {
            'mlp_64_acc': float(scores_mlp.mean()),
            'mlp_64_std': float(scores_mlp.std()),
            'mlp_128_acc': float(scores_mlp2.mean()),
            'mlp_128_std': float(scores_mlp2.std()),
            'random_acc': float(scores_random.mean()),
            'random_std': float(scores_random.std()),
        }
        
        p852_results[f'layer_{layer_idx}'] = {
            'linear_acc': float(scores_linear.mean()),
            'linear_std': float(scores_linear.std()),
            'mlp_acc': float(scores_mlp.mean()),
            'mlp_std': float(scores_mlp.std()),
            'nonlinear_gain': float(scores_mlp.mean() / max(scores_linear.mean(), 0.01)),
        }
        
        p853_results[f'layer_{layer_idx}'] = {
            'linear': float(scores_linear.mean()),
            'mlp_64': float(scores_mlp.mean()),
            'mlp_128': float(scores_mlp2.mean()),
        }
    
    # === P853: 逐层功能可解码性曲线 ===
    print(f"\n--- P853: Layer-wise Functional Decodability ---", flush=True)
    
    # 在所有层做MLP probe
    all_layer_scores = {}
    for layer_idx in range(0, n_layers, max(1, n_layers//10)):
        X = get_hidden_states_batch(model, tokenizer, device, all_texts, layer_idx)
        y = np.array(all_labels)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=42, early_stopping=True)
        scores = cross_val_score(clf, X_scaled, y, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), scoring='accuracy')
        
        all_layer_scores[layer_idx] = float(scores.mean())
        print(f"  Layer {layer_idx}: MLP acc = {scores.mean():.3f}", flush=True)
    
    # === P854: kernel PCA分析非线性功能结构 ===
    print(f"\n--- P854: Kernel PCA Nonlinear Structure ---", flush=True)
    
    from sklearn.decomposition import KernelPCA
    
    # 在Layer 0做kernel PCA
    X_layer0 = get_hidden_states_batch(model, tokenizer, device, all_texts, layer_idx=0)
    y = np.array(all_labels)
    
    # 线性PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(35, X_layer0.shape[0]-1))
    X_pca = pca.fit_transform(StandardScaler().fit_transform(X_layer0))
    
    # RBF kernel PCA
    kpca = KernelPCA(n_components=min(35, X_layer0.shape[0]-1), kernel='rbf', gamma=0.01, random_state=42)
    X_kpca = kpca.fit_transform(StandardScaler().fit_transform(X_layer0))
    
    # 用PCA和kernel PCA的特征分别做分类
    clf_test = LogisticRegression(max_iter=1000)
    
    # Linear PCA features
    scores_pca = cross_val_score(clf_test, X_pca[:, :10], y, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42))
    
    # Kernel PCA features
    scores_kpca = cross_val_score(clf_test, X_kpca[:, :10], y, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42))
    
    print(f"  Linear PCA (10 components): {scores_pca.mean():.3f}±{scores_pca.std():.3f}", flush=True)
    print(f"  Kernel PCA (10 components): {scores_kpca.mean():.3f}±{scores_kpca.std():.3f}", flush=True)
    print(f"  Kernel gain: {scores_kpca.mean()/max(scores_pca.mean(),0.01):.2f}x", flush=True)
    
    p854_results = {
        'pca_acc': float(scores_pca.mean()),
        'kpca_acc': float(scores_kpca.mean()),
        'kernel_gain': float(scores_kpca.mean() / max(scores_pca.mean(), 0.01)),
    }
    
    # 释放模型
    release_model(model)
    
    # 保存结果
    results = {
        'model': model_name,
        'd_model': d_model,
        'n_layers': n_layers,
        'n_samples': n_samples,
        'n_classes': len(func_type_names),
        'timestamp': datetime.now().isoformat(),
        
        'p851_mlp_classification': p851_results,
        'p852_linear_vs_mlp': p852_results,
        'p853_layer_decodability': all_layer_scores,
        'p854_kernel_pca': p854_results,
    }
    
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'phase_clxxxv')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f'{model_name}_results.json')
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_file}", flush=True)
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['glm4', 'qwen3', 'deepseek7b'])
    args = parser.parse_args()
    
    run_experiment(args.model)
