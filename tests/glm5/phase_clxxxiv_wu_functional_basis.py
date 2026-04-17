"""
Phase CLXXXIV: W_U作为功能基底 — 从输出矩阵反推语言结构
核心洞察: Qwen3的W_U-SV[0]是功能枢纽(polarity→SV[0]=94.88%)
问题: W_U的奇异向量是否定义了语言功能的"自然基底"?

测试:
P841: W_U top奇异向量的功能语义 — 每个SV对哪类功能对最敏感?
P842: W_U-SV对功能对的"分类能力" — 用SV投影能区分功能对吗?
P843: 训练方法与W_U功能结构的因果联系 — Qwen3 vs GLM4 vs DS7B
P844: W_U奇异向量的跨层稳定性 — 不同层W_U投影的一致性

Usage:
    python phase_clxxxiv_wu_functional_basis.py --model glm4
    python phase_clxxxiv_wu_functional_basis.py --model qwen3
    python phase_clxxxiv_wu_functional_basis.py --model deepseek7b
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


def get_function_pairs():
    """定义35个功能对，按5个功能类别"""
    return {
        'syntax': [
            ("The cat sits", "Cats sit"),
            ("She runs", "She run"),
            ("The big cat", "The cat big"),
            ("I have eaten", "I eat have"),
            ("He is running", "Him running is"),
            ("The cat that runs", "Cat the that runs"),
            ("She will go", "Will she go"),
        ],
        'semantic': [
            ("The cat sits", "The dog sits"),
            ("She is happy", "She is sad"),
            ("The house is big", "The house is small"),
            ("He loves music", "He hates music"),
            ("The king ruled", "The queen ruled"),
            ("I bought food", "I stole food"),
            ("The sun rises", "The sun sets"),
        ],
        'style': [
            ("I am happy", "I'm happy"),
            ("Do not go", "Don't go"),
            ("It is fine", "It's fine"),
            ("She will come", "She'll come"),
            ("I have done", "I've done"),
            ("Cannot do", "Can't do"),
            ("We are here", "We're here"),
        ],
        'tense': [
            ("She walks", "She walked"),
            ("He runs", "He ran"),
            ("I go", "I went"),
            ("They see", "They saw"),
            ("We eat", "We ate"),
            ("She writes", "She wrote"),
            ("He thinks", "He thought"),
        ],
        'polarity': [
            ("She is happy", "She is not happy"),
            ("I like it", "I don't like it"),
            ("He can go", "He cannot go"),
            ("This is good", "This is not good"),
            ("She will come", "She will not come"),
            ("They have food", "They have no food"),
            ("We know", "We do not know"),
        ],
    }


def get_hidden_states(model, tokenizer, device, text, layer_idx=0):
    """获取指定层的残差流"""
    tokens = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
    hs = to_numpy(outputs.hidden_states[layer_idx][0, -1])
    return hs


def run_experiment(model_name):
    """运行CLXXXIV实验"""
    print(f"\n{'='*60}", flush=True)
    print(f"Phase CLXXXIV: W_U as Functional Basis - {model_name}", flush=True)
    print(f"{'='*60}", flush=True)
    
    # 加载模型
    model, tokenizer, device = load_model(model_name)
    d_model = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    
    # 获取W_U
    W_U = to_numpy(model.lm_head.weight)  # [vocab, d_model]
    print(f"W_U shape: {W_U.shape}, d_model={d_model}, n_layers={n_layers}", flush=True)
    
    # W_U截断SVD
    from sklearn.decomposition import TruncatedSVD
    n_svd = min(200, d_model, W_U.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_svd, random_state=42)
    svd.fit(W_U)
    s_wu = svd.singular_values_
    Vt_wu = svd.components_  # [n_svd, d_model]
    
    print(f"W_U SVD: top-5 singular values = {s_wu[:5]}", flush=True)
    
    # 获取功能对
    pairs_dict = get_function_pairs()
    
    # === P841: W_U top奇异向量的功能语义 ===
    print(f"\n--- P841: W_U SV Functional Semantics ---", flush=True)
    
    # 对每个功能对，计算残差流差异在W_U奇异空间中的投影
    sv_sensitivity = np.zeros((n_svd, 5))  # [n_svd, 5 func_types]
    func_type_names = ['syntax', 'semantic', 'style', 'tense', 'polarity']
    
    for ft_idx, (func_type, pairs) in enumerate(pairs_dict.items()):
        for text_a, text_b in pairs:
            hs_a = get_hidden_states(model, tokenizer, device, text_a, layer_idx=0)
            hs_b = get_hidden_states(model, tokenizer, device, text_b, layer_idx=0)
            diff = hs_a - hs_b
            
            # 投影到W_U奇异空间
            proj = Vt_wu @ diff  # [n_svd]
            sv_sensitivity[:, ft_idx] += np.abs(proj)
        
        sv_sensitivity[:, ft_idx] /= len(pairs)
    
    # 每个SV对哪种功能最敏感
    dominant_func = np.argmax(sv_sensitivity, axis=1)
    dominant_sensitivity = np.max(sv_sensitivity, axis=1)
    
    p841_results = {}
    for sv_idx in range(min(20, n_svd)):
        ft = func_type_names[dominant_func[sv_idx]]
        sens = dominant_sensitivity[sv_idx]
        all_sens = sv_sensitivity[sv_idx].tolist()
        
        # 该SV的top words
        proj_vocab = svd.transform(W_U)  # [vocab, n_svd]
        sv_scores = proj_vocab[:, sv_idx]
        top_indices = np.argsort(np.abs(sv_scores))[-5:][::-1]
        top_words = []
        for idx in top_indices:
            try:
                word = tokenizer.decode([idx]).encode('ascii', 'replace').decode('ascii').strip()
            except:
                word = f"token_{idx}"
            top_words.append(word)
        
        p841_results[f'SV_{sv_idx}'] = {
            'dominant_function': ft,
            'sensitivity': float(sens),
            'all_sensitivities': {func_type_names[i]: float(sv_sensitivity[sv_idx, i]) for i in range(5)},
            'top_words': top_words,
        }
        if sv_idx < 10:
            print(f"  SV[{sv_idx}]: dominant={ft}, sens={sens:.4f}, words={top_words}", flush=True)
    
    # === P842: W_U-SV对功能对的分类能力 ===
    print(f"\n--- P842: W_U-SV Classification of Function Pairs ---", flush=True)
    
    # 收集所有功能对的差异
    all_diffs = []
    all_labels = []
    
    for ft_idx, (func_type, pairs) in enumerate(pairs_dict.items()):
        for text_a, text_b in pairs:
            hs_a = get_hidden_states(model, tokenizer, device, text_a, layer_idx=0)
            hs_b = get_hidden_states(model, tokenizer, device, text_b, layer_idx=0)
            all_diffs.append(hs_a - hs_b)
            all_labels.append(ft_idx)
    
    all_diffs = np.array(all_diffs)  # [35, d_model]
    all_labels = np.array(all_labels)
    
    # 投影到W_U top-k奇异空间
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    
    p842_results = {}
    for k in [5, 10, 20, 50]:
        # 投影到W_U top-k SV
        proj = all_diffs @ Vt_wu[:k].T  # [35, k]
        
        # 5-fold交叉验证分类
        clf = LogisticRegression(max_iter=1000)
        scores = cross_val_score(clf, proj, all_labels, cv=min(5, len(all_diffs)//5), scoring='accuracy')
        
        p842_results[f'k={k}'] = {
            'accuracy': float(np.mean(scores)),
            'std': float(np.std(scores)),
        }
        print(f"  k={k}: accuracy={np.mean(scores):.3f}±{np.std(scores):.3f}", flush=True)
    
    # 随机基线: 用随机正交基替代W_U SV
    rng = np.random.default_rng(42)
    V_rand = rng.standard_normal((50, d_model))
    Q_rand, _ = np.linalg.qr(V_rand.T)
    Q_rand = Q_rand.T[:50]
    
    proj_rand = all_diffs @ Q_rand.T
    clf_rand = LogisticRegression(max_iter=1000)
    scores_rand = cross_val_score(clf_rand, proj_rand, all_labels, cv=min(5, len(all_diffs)//5), scoring='accuracy')
    print(f"  Random baseline: accuracy={np.mean(scores_rand):.3f}±{np.std(scores_rand):.3f}", flush=True)
    p842_results['random_baseline'] = {
        'accuracy': float(np.mean(scores_rand)),
        'std': float(np.std(scores_rand)),
    }
    
    # === P843: 训练方法与W_U功能结构 ===
    print(f"\n--- P843: Training Method & W_U Functional Structure ---", flush=True)
    
    # 计算每类功能在W_U top SV中的集中度
    p843_results = {}
    for ft_idx, func_type in enumerate(func_type_names):
        # 该功能类型在top-1 SV中的能量占比
        energy_in_sv0 = sv_sensitivity[0, ft_idx] / np.sum(sv_sensitivity[:, ft_idx])
        # 该功能类型在top-5 SV中的能量占比
        energy_in_top5 = np.sum(sv_sensitivity[:5, ft_idx]) / np.sum(sv_sensitivity[:, ft_idx])
        # 该功能类型在top-20 SV中的能量占比
        energy_in_top20 = np.sum(sv_sensitivity[:20, ft_idx]) / np.sum(sv_sensitivity[:, ft_idx])
        
        p843_results[func_type] = {
            'energy_in_sv0': float(energy_in_sv0),
            'energy_in_top5': float(energy_in_top5),
            'energy_in_top20': float(energy_in_top20),
        }
        print(f"  {func_type}: SV0={energy_in_sv0:.3f}, top5={energy_in_top5:.3f}, top20={energy_in_top20:.3f}", flush=True)
    
    # === P844: W_U奇异向量的跨层稳定性 ===
    print(f"\n--- P844: Cross-Layer Stability of W_U Projection ---", flush=True)
    
    # 检查不同层的残差流在W_U奇异空间中的分布是否一致
    layer_projections = {}
    for layer_idx in [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]:
        # 用一组测试句子获取残差流
        test_texts = ["The cat sits", "She is happy", "I walk to the store"]
        projections = []
        for text in test_texts:
            hs = get_hidden_states(model, tokenizer, device, text, layer_idx=layer_idx)
            proj = Vt_wu[:20] @ hs  # [20]
            projections.append(proj)
        
        layer_projections[f'layer_{layer_idx}'] = {
            'mean_proj_norm': float(np.mean([np.linalg.norm(p) for p in projections])),
            'proj_per_sv': [float(np.mean([abs(p[i]) for p in projections])) for i in range(20)],
        }
        print(f"  Layer {layer_idx}: mean_proj_norm={layer_projections[f'layer_{layer_idx}']['mean_proj_norm']:.4f}", flush=True)
    
    # 检查SV0的投影是否在各层一致
    sv0_projections = [layer_projections[f'layer_{l}']['proj_per_sv'][0] for l in [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]]
    print(f"  SV0 projection across layers: {[f'{x:.4f}' for x in sv0_projections]}", flush=True)
    
    # 释放模型
    release_model(model)
    
    # 保存结果
    results = {
        'model': model_name,
        'd_model': d_model,
        'n_layers': n_layers,
        'timestamp': datetime.now().isoformat(),
        
        'p841_sv_functional_semantics': p841_results,
        'p842_classification': p842_results,
        'p843_training_structure': p843_results,
        'p844_cross_layer': layer_projections,
        'p844_sv0_across_layers': sv0_projections,
    }
    
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'phase_clxxxiv')
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
