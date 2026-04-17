"""
Phase CLXXXVI: Probe真实性验证与残差流传播动力学
核心问题: CLXXXV的线性probe 79-90%准确率是真实的还是维度诅咒?

测试:
P861: 维度诅咒对照 — 降维后再probe，如果准确率随维度下降急剧降低→维度诅咒
P862: 配对t检验验证 — 对同一功能对的两成员，probe预测是否一致?
P863: 残差流传播追踪 — 功能差异向量在层间如何传播?
P864: 残差连接vs注意力/FFN对功能信号的贡献

Usage:
    python phase_clxxxvi_probe_validation.py --model glm4
    python phase_clxxxvi_probe_validation.py --model qwen3
    python phase_clxxxvi_probe_validation.py --model deepseek7b
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


def run_experiment(model_name):
    print(f"\n{'='*60}", flush=True)
    print(f"Phase CLXXXVI: Probe Validation & Propagation Dynamics - {model_name}", flush=True)
    print(f"{'='*60}", flush=True)
    
    model, tokenizer, device = load_model(model_name)
    d_model = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    
    pairs_dict = get_function_pairs_expanded()
    func_type_names = list(pairs_dict.keys())
    
    # 准备数据
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
    
    y = np.array(all_labels)
    pair_ids = np.array(all_pair_ids)
    
    # === P861: 维度诅咒对照 ===
    print(f"\n--- P861: Dimension Curse Control ---", flush=True)
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 在Layer 0获取残差流
    all_hs_layer0 = []
    for text in all_texts:
        tokens = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
        hs = to_numpy(outputs.hidden_states[0][0, -1])
        all_hs_layer0.append(hs)
    X_full = np.array(all_hs_layer0)
    
    # 降维后probe
    p861_results = {}
    max_dim = min(69, d_model)  # n_samples-1
    for n_dim in [5, 10, 20, 50, max_dim]:
        if n_dim > max_dim:
            n_dim = max_dim
        
        if n_dim < d_model:
            pca = PCA(n_components=n_dim, random_state=42)
            X_reduced = pca.fit_transform(StandardScaler().fit_transform(X_full))
        else:
            X_reduced = StandardScaler().fit_transform(X_full)
        
        clf = LogisticRegression(max_iter=2000, C=1.0)
        scores = cross_val_score(clf, X_reduced, y, cv=cv, scoring='accuracy')
        
        # 随机标签对照
        rng = np.random.default_rng(42)
        y_shuffle = y.copy()
        rng.shuffle(y_shuffle)
        scores_rand = cross_val_score(clf, X_reduced, y_shuffle, cv=cv, scoring='accuracy')
        
        p861_results[f'dim_{n_dim}'] = {
            'accuracy': float(scores.mean()),
            'std': float(scores.std()),
            'random_acc': float(scores_rand.mean()),
            'real_vs_random': float(scores.mean() / max(scores_rand.mean(), 0.01)),
        }
        print(f"  dim={n_dim}: acc={scores.mean():.3f}±{scores.std():.3f}, random={scores_rand.mean():.3f}, ratio={scores.mean()/max(scores_rand.mean(),0.01):.1f}x", flush=True)
    
    # === P862: 配对验证 ===
    print(f"\n--- P862: Pair Validation ---", flush=True)
    
    # 同一功能对的两成员应该被分类到同一类
    # 如果probe真的学到了功能模式，配对成员的分类应该一致
    X_scaled = StandardScaler().fit_transform(X_full)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X_scaled, y)
    predictions = clf.predict(X_scaled)
    
    # 检查配对一致性
    pair_consistency = 0
    n_pairs = len(set(pair_ids))
    for pid in range(n_pairs):
        idx_a = np.where(pair_ids == pid)[0][0]
        idx_b = np.where(pair_ids == pid)[0][1]
        if predictions[idx_a] == predictions[idx_b] == y[idx_a]:
            pair_consistency += 1
    
    consistency_rate = pair_consistency / n_pairs
    print(f"  Pair consistency: {pair_consistency}/{n_pairs} = {consistency_rate:.3f}", flush=True)
    
    # 随机对照: 随机预测的配对一致性
    rng2 = np.random.default_rng(123)
    n_sim = 1000
    random_consistency = 0
    for _ in range(n_sim):
        for pid in range(n_pairs):
            idx_a = np.where(pair_ids == pid)[0][0]
            idx_b = np.where(pair_ids == pid)[0][1]
            pred_a = rng2.integers(0, 5)
            pred_b = rng2.integers(0, 5)
            if pred_a == pred_b == y[idx_a]:
                random_consistency += 1
    random_rate = random_consistency / (n_sim * n_pairs)
    print(f"  Random consistency: {random_rate:.3f}", flush=True)
    print(f"  Real/Random ratio: {consistency_rate/random_rate:.1f}x", flush=True)
    
    p862_results = {
        'pair_consistency': float(consistency_rate),
        'random_consistency': float(random_rate),
        'ratio': float(consistency_rate / random_rate),
    }
    
    # === P863: 残差流传播追踪 ===
    print(f"\n--- P863: Residual Stream Propagation Tracking ---", flush=True)
    
    # 追踪功能差异向量在层间的范数和方向变化
    test_pairs = [
        ("syntax", "The cat sits", "Cats sit"),
        ("semantic", "She is very happy now", "She is very sad now"),
        ("polarity", "She is happy now", "She is not happy now"),
    ]
    
    p863_results = {}
    for func_name, text_a, text_b in test_pairs:
        diffs = []
        norms = []
        cos_with_layer0 = []
        
        for layer_idx in range(n_layers):
            hs_a = get_hs_at_layer(model, tokenizer, device, text_a, layer_idx)
            hs_b = get_hs_at_layer(model, tokenizer, device, text_b, layer_idx)
            diff = hs_a - hs_b
            diffs.append(diff)
            norms.append(np.linalg.norm(diff))
            
            if layer_idx > 0:
                cos = np.dot(diff, diffs[0]) / (np.linalg.norm(diff) * np.linalg.norm(diffs[0]) + 1e-10)
                cos_with_layer0.append(float(cos))
            else:
                cos_with_layer0.append(1.0)
        
        p863_results[func_name] = {
            'norms': [float(n) for n in norms],
            'cos_with_layer0': cos_with_layer0,
            'norm_ratio_last_first': float(norms[-1] / max(norms[0], 1e-10)),
        }
        
        print(f"  {func_name}: norm0={norms[0]:.4f}, norm_last={norms[-1]:.4f}, ratio={norms[-1]/max(norms[0],1e-10):.1f}x", flush=True)
        print(f"    cos(L0,L_mid)={cos_with_layer0[n_layers//2]:.3f}, cos(L0,L_last)={cos_with_layer0[-1]:.3f}", flush=True)
    
    # === P864: 残差连接vs注意力/FFN贡献 ===
    print(f"\n--- P864: Residual vs Attn/FFN Contribution ---", flush=True)
    
    # 对于同一对文本，计算每层残差连接和Attn/FFN对功能差异的贡献
    text_a, text_b = "She is happy now", "She is not happy now"
    
    # 获取每层的残差流
    hs_a_layers = []
    hs_b_layers = []
    for layer_idx in range(n_layers):
        hs_a_layers.append(get_hs_at_layer(model, tokenizer, device, text_a, layer_idx))
        hs_b_layers.append(get_hs_at_layer(model, tokenizer, device, text_b, layer_idx))
    
    # 功能差异的层间变化
    diff_layers = [hs_a - hs_b for hs_a, hs_b in zip(hs_a_layers, hs_b_layers)]
    
    # 计算残差连接保留了多少差异
    residual_retention = []
    for l in range(1, len(diff_layers)):
        diff_change = diff_layers[l] - diff_layers[l-1]
        residual_retention.append({
            'layer': l,
            'diff_norm': float(np.linalg.norm(diff_layers[l])),
            'change_norm': float(np.linalg.norm(diff_change)),
            'change_ratio': float(np.linalg.norm(diff_change) / max(np.linalg.norm(diff_layers[l]), 1e-10)),
        })
    
    # 打印关键层
    for r in residual_retention[::max(1, len(residual_retention)//5)]:
        print(f"  Layer {r['layer']}: diff_norm={r['diff_norm']:.4f}, change_norm={r['change_norm']:.4f}, ratio={r['change_ratio']:.4f}", flush=True)
    
    p864_results = {
        'residual_retention': residual_retention,
    }
    
    release_model(model)
    
    results = {
        'model': model_name,
        'd_model': d_model,
        'n_layers': n_layers,
        'n_samples': len(all_texts),
        'timestamp': datetime.now().isoformat(),
        
        'p861_dimension_curse': p861_results,
        'p862_pair_validation': p862_results,
        'p863_propagation': p863_results,
        'p864_residual_contribution': p864_results,
    }
    
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'phase_clxxxvi')
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f'{model_name}_results.json')
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_file}", flush=True)
    
    return results


def get_hs_at_layer(model, tokenizer, device, text, layer_idx):
    tokens = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
    return to_numpy(outputs.hidden_states[layer_idx][0, -1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['glm4', 'qwen3', 'deepseek7b'])
    args = parser.parse_args()
    
    run_experiment(args.model)
